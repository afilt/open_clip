"""
Mid-training pathology evaluation hooks for open_clip fine-tuning.

Provides two evaluations run after each epoch (on rank-0 only):

  run_tcga_eval(model_visual, args, device)
      → TCGA-UT 31-class linear probe (catastrophic-forgetting monitor)
        Returns {'tcga_acc': float, 'tcga_bal_acc': float}

  run_scorpion_eval(model_visual, args, device)
      → SCORPION scanner-invariance retrieval (robustness metric)
        Returns {'scorpion_r1': float, 'scorpion_map': float}

  run_pathology_evals(model, args, completed_epoch, tb_writer)
      → Calls both, logs results, writes to tensorboard / wandb.

Both functions use model.visual (TimmModel) as the feature extractor.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Pathology-domain normalisation (same values used during training)
_PIXEL_MEAN = [0.707223, 0.578729, 0.703617]
_PIXEL_STD  = [0.211883, 0.230117, 0.177517]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_PIXEL_MEAN, std=_PIXEL_STD),
])


# ---------------------------------------------------------------------------
# Generic tile dataset (used by both evals)
# ---------------------------------------------------------------------------

class _TileDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _EVAL_TRANSFORM(Image.open(self.paths[idx]).convert("RGB"))


@torch.no_grad()
def _extract_features(
    model_visual: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return L2-normalised feature matrix (N, D)."""
    ds = _TileDataset(paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    use_amp = device.type == "cuda"
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            f = model_visual(batch)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats)


# ---------------------------------------------------------------------------
# TCGA-UT linear probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tcga_eval(
    model_visual: torch.nn.Module,
    data_root: str,
    device: torch.device,
    tile_batch_size: int = 32,
    max_tiles: int = 50,
) -> dict:
    """31-class cancer-type linear probe on TCGA-UT.

    Performs a stratified 80/20 slide-level split, extracts mean-pooled
    features, trains a LogisticRegression, and reports accuracy + balanced
    accuracy.

    Returns
    -------
    dict with keys ``tcga_acc`` and ``tcga_bal_acc`` (float, 0–1).
    Empty dict if ``data_root`` has no slides.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logging.warning("scikit-learn not installed; skipping TCGA-UT eval.")
        return {}

    import json as _json

    root = Path(data_root)
    label2idx: dict[str, int] = {}
    rows: list[dict] = []

    for slide_dir in sorted(root.iterdir()):
        if not slide_dir.is_dir():
            continue
        jsons = list(slide_dir.glob("*.json"))
        if not jsons:
            continue
        with open(jsons[0]) as f:
            label = _json.load(f)["label"].replace(" ", "_")
        if label not in label2idx:
            label2idx[label] = len(label2idx)
        rows.append({"slide_dir": slide_dir, "label_idx": label2idx[label]})

    if not rows:
        return {}

    # Stratified 80/20 split
    rng = random.Random(42)
    by_label: dict[int, list] = {}
    for r in rows:
        by_label.setdefault(r["label_idx"], []).append(r)
    train_rows, test_rows = [], []
    for grp in by_label.values():
        rng.shuffle(grp)
        n_test = max(1, round(len(grp) * 0.2))
        test_rows.extend(grp[:n_test])
        train_rows.extend(grp[n_test:])

    model_visual.eval()

    def _slide_feat(slide_dir: Path) -> Optional[np.ndarray]:
        tile_paths = sorted(slide_dir.glob("*.jpg"))
        if not tile_paths:
            return None
        if max_tiles and len(tile_paths) > max_tiles:
            tile_paths = random.sample(tile_paths, max_tiles)
        feats = _extract_features(model_visual, tile_paths, device, tile_batch_size)
        return feats.mean(0).numpy()

    def _build_Xy(split_rows: list[dict]):
        X, y = [], []
        for r in split_rows:
            feat = _slide_feat(r["slide_dir"])
            if feat is not None:
                X.append(feat)
                y.append(r["label_idx"])
        return np.array(X), np.array(y)

    X_tr, y_tr = _build_Xy(train_rows)
    X_te, y_te = _build_Xy(test_rows)
    if not len(X_tr) or not len(X_te):
        return {}

    sc = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    preds = clf.predict(sc.transform(X_te))

    return {
        "tcga_acc":     float(accuracy_score(y_te, preds)),
        "tcga_bal_acc": float(balanced_accuracy_score(y_te, preds)),
    }


# ---------------------------------------------------------------------------
# SCORPION retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_scorpion_eval(
    model_visual: torch.nn.Module,
    data_root: str,
    device: torch.device,
    batch_size: int = 32,
    max_slides: Optional[int] = None,
) -> dict:
    """Scanner-invariance retrieval on SCORPION (R@1, mAP).

    Expected layout::

        <data_root>/slide_{i}/sample_{j}/{SCANNER}.jpg

    Positives for a query = images of the same (slide, sample) from
    different scanners.

    Returns
    -------
    dict with keys ``scorpion_r1`` and ``scorpion_map`` (float, 0–1).
    """
    SCANNERS = ["AT2", "DP200", "GT450", "P1000", "Philips"]
    root = Path(data_root)

    items: list[dict] = []
    slide_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("slide_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if max_slides:
        slide_dirs = slide_dirs[:max_slides]

    for slide_dir in slide_dirs:
        s_idx = int(slide_dir.name.split("_")[1])
        for sample_dir in sorted(slide_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sa_idx = int(sample_dir.name.split("_")[1])
            for sc in SCANNERS:
                p = sample_dir / f"{sc}.jpg"
                if p.exists():
                    items.append({
                        "path": p,
                        "group_id": s_idx * 10_000 + sa_idx,
                    })

    if not items:
        return {}

    model_visual.eval()
    all_paths = [it["path"] for it in items]
    features = _extract_features(
        model_visual, all_paths, device, batch_size
    ).numpy()
    group_arr = np.array([it["group_id"] for it in items])

    N = len(features)
    sim = (torch.from_numpy(features) @ torch.from_numpy(features).T).numpy()
    np.fill_diagonal(sim, -2.0)
    ranked = np.argsort(-sim, axis=1)
    pos_ranked = group_arr[ranked] == group_arr[:, None]
    self_mask = ranked == np.arange(N)[:, None]
    pos_ranked &= ~self_mask

    r1 = pos_ranked[:, 0].astype(float).mean()
    n_pos = pos_ranked.sum(axis=1)
    valid = n_pos > 0
    cumhits = pos_ranked.cumsum(axis=1).astype(float)
    ranks = np.arange(1, N + 1, dtype=float)
    ap = np.where(
        valid,
        (cumhits / ranks * pos_ranked).sum(axis=1) / np.maximum(n_pos, 1),
        0.0,
    )
    return {"scorpion_r1": float(r1), "scorpion_map": float(ap.mean())}


# ---------------------------------------------------------------------------
# Combined entry point called from main.py
# ---------------------------------------------------------------------------

def run_pathology_evals(
    model,
    args,
    completed_epoch: int,
    tb_writer=None,
) -> None:
    """Run TCGA-UT and/or SCORPION eval and log results.

    Parameters
    ----------
    model : CustomTextCLIP (unwrapped, not DDP)
    args  : parsed args from open_clip_train.params
    completed_epoch : int (1-indexed)
    tb_writer : tensorboard SummaryWriter or None
    """
    try:
        import wandb as _wandb
    except ImportError:
        _wandb = None

    device = next(model.parameters()).device
    model_visual = model.visual
    model_visual.eval()

    batch_size = getattr(args, "eval_batch_size", 32)
    max_tiles  = getattr(args, "eval_max_tiles", 50)
    metrics: dict[str, float] = {}

    if getattr(args, "eval_tcga_root", None):
        logging.info("[pathology eval] Running TCGA-UT linear probe …")
        tcga = run_tcga_eval(
            model_visual,
            args.eval_tcga_root,
            device,
            tile_batch_size=batch_size,
            max_tiles=max_tiles,
        )
        metrics.update(tcga)
        if tcga:
            logging.info(
                f"  TCGA-UT  acc={tcga.get('tcga_acc', 0):.4f}  "
                f"bal_acc={tcga.get('tcga_bal_acc', 0):.4f}"
            )

    if getattr(args, "eval_scorpion_root", None):
        logging.info("[pathology eval] Running SCORPION retrieval …")
        scorp = run_scorpion_eval(
            model_visual,
            args.eval_scorpion_root,
            device,
            batch_size=batch_size,
        )
        metrics.update(scorp)
        if scorp:
            logging.info(
                f"  SCORPION R@1={scorp.get('scorpion_r1', 0):.4f}  "
                f"mAP={scorp.get('scorpion_map', 0):.4f}"
            )

    if not metrics:
        return

    # Tensorboard
    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"pathology_eval/{k}", v, completed_epoch)

    # Wandb
    if _wandb is not None and getattr(args, "wandb", False):
        _wandb.log(
            {f"pathology_eval/{k}": v for k, v in metrics.items()},
            step=completed_epoch,
        )

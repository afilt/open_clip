#!/usr/bin/env python3
"""
Scanner-invariance retrieval evaluation on the SCORPION dataset.

Dataset structure
-----------------
<root>/
  slide_{i}/
    sample_{j}/
      AT2.jpg
      DP200.jpg
      GT450.jpg
      P1000.jpg
      Philips.jpg

Each (slide_i, sample_j) is the same physical tissue location scanned by 5
different slide scanners.  The task is to retrieve the same location from
other scanners — a direct measure of scanner-invariance (= robustness).

Two evaluation modes
--------------------
  Full image (default)
      Each 1024×1024 image is resized/cropped to 224×224 and treated as a
      single query/gallery item.

  Grid mode (--grid)
      Each image is tiled into a 4×4 grid of 224×224 crops.
      Positives = same grid position, different scanner.  Harder task.

Retrieval gallery
-----------------
  Standard (default)
      Global gallery — all items in the dataset.

  Hard (--hard)
      Restricted gallery — for each query:
        positives = same (slide, sample) from the 4 other scanners
        negatives = other samples from the *same* slide × scanner variant
      Negatives are visually similar (same tissue section) so this is
      significantly harder.

Metrics
-------
  R@1, R@5, R@10, mAP, MRR — global + per-scanner breakdown

Usage
-----
  # Baseline (no LoRA):
  python eval_scorpion.py \\
      --data_root /Users/afiliot/Desktop/scorpion \\
      --baseline

  # Fine-tuned checkpoint:
  python eval_scorpion.py \\
      --data_root /Users/afiliot/Desktop/scorpion \\
      --checkpoint checkpoints/pathgen_clip/best_checkpoint.pt

  # Hard mode with grid tiling:
  python eval_scorpion.py ... --hard --grid

Adapted from:
  /Users/afiliot/Desktop/libs/constrastive-plism/eval_scorpion_retrieval.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Resolve open_clip from the repo root
# ---------------------------------------------------------------------------
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))

from open_clip.lora_utils import load_visual_from_open_clip_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIXEL_MEAN = [0.707223, 0.578729, 0.703617]
PIXEL_STD  = [0.211883, 0.230117, 0.177517]
SCANNERS   = ["AT2", "DP200", "GT450", "P1000", "Philips"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _iter_slide_sample_dirs(root: Path, max_slides: Optional[int] = None):
    slide_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("slide_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if max_slides is not None:
        slide_dirs = slide_dirs[:max_slides]
    for slide_dir in slide_dirs:
        s_idx = int(slide_dir.name.split("_")[1])
        sample_dirs = sorted(
            [p for p in slide_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        for sample_dir in sample_dirs:
            yield s_idx, int(sample_dir.name.split("_")[1]), sample_dir


class ScorpionDataset(Dataset):
    """One item per (slide, sample, scanner): full 1024×1024 image → 224×224.

    group_id  = slide_idx × 10_000 + sample_idx
    variant_id = slide_idx × n_scanners + scanner_idx
    """

    def __init__(
        self, root: str | Path, transform, max_slides: Optional[int] = None
    ) -> None:
        self.transform = transform
        self.items: list[dict] = []
        for s_idx, sa_idx, sample_dir in _iter_slide_sample_dirs(Path(root), max_slides):
            for scanner in SCANNERS:
                p = sample_dir / f"{scanner}.jpg"
                if p.exists():
                    self.items.append({
                        "path":       p,
                        "slide_idx":  s_idx,
                        "sample_idx": sa_idx,
                        "scanner":    scanner,
                        "group_id":   s_idx * 10_000 + sa_idx,
                        "variant_id": s_idx * len(SCANNERS) + SCANNERS.index(scanner),
                    })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        img  = Image.open(item["path"]).convert("RGB")
        return self.transform(img), item["group_id"], item["scanner"]


class ScorpionGridDataset(Dataset):
    """4×4 grid of 224×224 tiles from each 1024×1024 SCORPION image.

    Total: n_images × 16 items.
    Positives = same grid position across scanners.
    """

    GRID = 4
    TILE = 224

    def __init__(
        self, root: str | Path, transform, max_slides: Optional[int] = None
    ) -> None:
        self.transform = transform
        self.items: list[dict] = []
        g, t = self.GRID, self.TILE
        # Evenly-spaced start positions (last = 1024 − 224 = 800)
        self._starts = [round(i * (1024 - t) / (g - 1)) for i in range(g)]

        for s_idx, sa_idx, sample_dir in _iter_slide_sample_dirs(Path(root), max_slides):
            for scanner in SCANNERS:
                p = sample_dir / f"{scanner}.jpg"
                if p.exists():
                    for gr in range(g):
                        for gc in range(g):
                            self.items.append({
                                "path":       p,
                                "slide_idx":  s_idx,
                                "sample_idx": sa_idx,
                                "scanner":    scanner,
                                "grid_row":   gr,
                                "grid_col":   gc,
                                "group_id": (
                                    s_idx * 10_000 * g * g
                                    + sa_idx * g * g
                                    + gr * g + gc
                                ),
                                "variant_id": (
                                    (s_idx * 10_000 + sa_idx) * len(SCANNERS)
                                    + SCANNERS.index(scanner)
                                ),
                            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        img  = Image.open(item["path"]).convert("RGB")
        x0   = self._starts[item["grid_col"]]
        y0   = self._starts[item["grid_row"]]
        tile = img.crop((x0, y0, x0 + self.TILE, y0 + self.TILE))
        return self.transform(tile), item["group_id"], item["scanner"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_features(
    backbone: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return L2-normalised features, group_ids, and scanner names."""
    backbone.eval()
    use_amp = device.type == "cuda"
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
    )

    all_feats, all_groups, all_scanners = [], [], []
    for imgs, group_ids, scanner_names in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            feats = backbone(imgs)
        feats = F.normalize(feats.float(), dim=-1)
        all_feats.append(feats.cpu().numpy())
        all_groups.extend(group_ids.tolist())
        all_scanners.extend(scanner_names)

    return np.vstack(all_feats), all_groups, all_scanners


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics_global(
    features: np.ndarray,
    group_ids: list[int],
    scanners: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 512,
) -> dict:
    """Standard global-gallery retrieval metrics (chunked for memory efficiency).

    Positives for query i = all items with the same group_id (excluding self).
    """
    N         = len(features)
    group_arr = np.array(group_ids)
    feat_t    = torch.from_numpy(features)
    ranks_arr = np.arange(1, N + 1, dtype=np.float32)

    r_at_k_chunks: dict[int, list] = {k: [] for k in ks}
    ap_chunks:  list[np.ndarray] = []
    mrr_chunks: list[np.ndarray] = []
    per_scanner_r1:  dict[str, list[float]] = defaultdict(list)
    per_scanner_map: dict[str, list[float]] = defaultdict(list)

    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        sim   = (feat_t[q_start:q_end] @ feat_t.T).numpy()   # (C, N)
        np.fill_diagonal(sim[:, q_start:q_end], -2.0)         # exclude self

        ranked     = np.argsort(-sim, axis=1)                  # (C, N)
        pos_ranked = group_arr[ranked] == group_arr[q_start:q_end, None]
        self_mask  = ranked == np.arange(q_start, q_end)[:, None]
        pos_ranked &= ~self_mask

        n_pos_q = pos_ranked.sum(axis=1)
        valid   = n_pos_q > 0

        for k in ks:
            r_at_k_chunks[k].append(
                pos_ranked[:, :k].any(axis=1).astype(np.float32)
            )

        cumhits   = pos_ranked.cumsum(axis=1).astype(np.float32)
        prec_at_r = cumhits / ranks_arr[None, :]
        ap = np.where(
            valid,
            (prec_at_r * pos_ranked).sum(axis=1) / np.maximum(n_pos_q, 1),
            0.0,
        )
        ap_chunks.append(ap)

        first_hit = pos_ranked.argmax(axis=1)
        mrr = np.where(valid, 1.0 / (first_hit + 1).astype(np.float32), 0.0)
        mrr_chunks.append(mrr)

        r1_chunk = pos_ranked[:, 0]
        for qi in range(q_end - q_start):
            if not valid[qi]:
                continue
            sc = scanners[q_start + qi]
            per_scanner_r1[sc].append(float(r1_chunk[qi]))
            per_scanner_map[sc].append(float(ap[qi]))

    return {
        **{f"R@{k}": float(np.concatenate(r_at_k_chunks[k]).mean()) for k in ks},
        "mAP": float(np.concatenate(ap_chunks).mean()),
        "MRR": float(np.concatenate(mrr_chunks).mean()),
        "per_scanner": {
            sc: {
                "R@1": float(np.mean(per_scanner_r1[sc])),
                "mAP": float(np.mean(per_scanner_map[sc])),
            }
            for sc in sorted(per_scanner_r1)
        },
    }


def compute_retrieval_metrics_restricted(
    features: np.ndarray,
    group_ids: list[int],
    variant_ids: list[int],
    scanners: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Hard restricted-gallery retrieval.

    For each query (group g, variant v):
      gallery  = {items in variant v, except self} ∪ {items in group g, except self}
      positives = items in group g that are NOT from variant v
    """
    N         = len(features)
    group_arr = np.array(group_ids, dtype=np.int64)

    variant_to_idx: dict[int, list[int]] = defaultdict(list)
    group_to_idx:   dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        variant_to_idx[variant_ids[i]].append(i)
        group_to_idx[group_ids[i]].append(i)

    r_at_k_all: dict[int, list[float]] = {k: [] for k in ks}
    ap_all, mrr_all = [], []
    per_scanner_r1:  dict[str, list[float]] = defaultdict(list)
    per_scanner_map: dict[str, list[float]] = defaultdict(list)

    for q in tqdm(range(N), desc="Restricted retrieval"):
        v, g = variant_ids[q], group_ids[q]

        gallery = np.array(
            sorted((set(variant_to_idx[v]) | set(group_to_idx[g])) - {q}),
            dtype=np.int64,
        )
        if len(gallery) == 0:
            continue

        pos_mask = (group_arr[gallery] == g)
        n_pos    = int(pos_mask.sum())
        if n_pos == 0:
            continue

        sim     = features[q] @ features[gallery].T
        ranked  = np.argsort(-sim)
        pos_r   = pos_mask[ranked]

        for k in ks:
            r_at_k_all[k].append(float(pos_r[:k].any()))

        cumhits = pos_r.cumsum().astype(np.float32)
        ranks_l = np.arange(1, len(gallery) + 1, dtype=np.float32)
        ap = float((cumhits / ranks_l * pos_r).sum() / n_pos)
        ap_all.append(ap)
        mrr_all.append(1.0 / (int(pos_r.argmax()) + 1))

        sc = scanners[q]
        per_scanner_r1[sc].append(float(pos_r[0]))
        per_scanner_map[sc].append(ap)

    return {
        **{f"R@{k}": float(np.mean(r_at_k_all[k])) for k in ks},
        "mAP": float(np.mean(ap_all)),
        "MRR": float(np.mean(mrr_all)),
        "per_scanner": {
            sc: {
                "R@1": float(np.mean(per_scanner_r1[sc])),
                "mAP": float(np.mean(per_scanner_map[sc])),
            }
            for sc in sorted(per_scanner_r1)
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _print_results(results: dict, mode_str: str) -> None:
    sep = "=" * 55
    lines = [
        sep,
        f"  Mode  : {mode_str}",
        f"  R@1   : {results['R@1']:.4f}  ({results['R@1']*100:.1f}%)",
        f"  R@5   : {results['R@5']:.4f}  ({results['R@5']*100:.1f}%)",
        f"  R@10  : {results['R@10']:.4f}  ({results['R@10']*100:.1f}%)",
        f"  mAP   : {results['mAP']:.4f}",
        f"  MRR   : {results['MRR']:.4f}",
        "  Per-scanner (query scanner → R@1 / mAP):",
        *[
            f"    {sc:<10}  R@1={m['R@1']:.3f}   mAP={m['mAP']:.3f}"
            for sc, m in results["per_scanner"].items()
        ],
        sep,
    ]
    print("\n".join(lines))


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ------------------------------------------------------------------ model
    if args.baseline or args.checkpoint is None:
        print("Loading base H0-mini (no LoRA) …")
        backbone, _ = load_visual_from_open_clip_checkpoint(
            checkpoint_path=None,
            model_name=args.open_clip_model,
            apply_lora=False,
            device=device,
        )
    else:
        print(f"Loading open_clip checkpoint: {args.checkpoint}")
        backbone, _ = load_visual_from_open_clip_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=args.open_clip_model,
            apply_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_vision_target_modules=args.lora_vision_target_modules,
            lora_text_target_modules=args.lora_text_target_modules,
            device=device,
        )

    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])

    # ------------------------------------------------------------------ data
    if args.grid:
        dataset   = ScorpionGridDataset(args.data_root, transform, args.max_slides)
        mode_str  = "grid 4×4 (224 px tiles)"
    else:
        dataset   = ScorpionDataset(args.data_root, transform, args.max_slides)
        mode_str  = "full image → 224 px"

    n_slides  = len({it["slide_idx"]              for it in dataset.items})
    n_samples = len({(it["slide_idx"], it["sample_idx"]) for it in dataset.items})
    print(
        f"SCORPION [{mode_str}]: {n_slides} slides, "
        f"{n_samples} (slide,sample) positions, "
        f"{len(dataset)} items ({len(SCANNERS)} scanners)"
    )

    # ---------------------------------------------------- feature extraction
    features, group_ids, scanners = extract_all_features(
        backbone, dataset, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Feature matrix: {features.shape}")

    # ------------------------------------------------------- metrics
    variant_ids = [it["variant_id"] for it in dataset.items]

    if args.hard:
        print("Computing restricted-gallery metrics …")
        results = compute_retrieval_metrics_restricted(
            features, group_ids, variant_ids, scanners
        )
        gallery_str = "restricted (same-slide negatives)"
    else:
        print("Computing global-gallery metrics …")
        results = compute_retrieval_metrics_global(
            features, group_ids, scanners
        )
        gallery_str = "global"

    _print_results(results, f"{mode_str} | {gallery_str}")

    # Save results
    if args.output_json:
        import json as _json
        out = {
            "R@1":  results["R@1"],
            "R@5":  results["R@5"],
            "R@10": results["R@10"],
            "mAP":  results["mAP"],
            "MRR":  results["MRR"],
            "per_scanner": results["per_scanner"],
            "checkpoint": args.checkpoint,
            "baseline": args.baseline,
            "grid": args.grid,
            "hard": args.hard,
        }
        with open(args.output_json, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scanner-invariance retrieval evaluation on SCORPION",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        default="/Users/afiliot/Desktop/scorpion",
    )
    p.add_argument("--checkpoint", default=None,
                   help="open_clip checkpoint (epoch_*.pt) from training")
    p.add_argument("--baseline", action="store_true",
                   help="Evaluate base H0-mini without LoRA")
    p.add_argument("--open_clip_model", default="H0-mini-BiomedBERT",
                   help="Registered open_clip model name (see model_configs/)")
    # LoRA args (only used when loading a checkpoint)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_vision_target_modules", nargs="+", default=["qkv"])
    p.add_argument("--lora_text_target_modules", nargs="+",
                   default=["query", "key", "value"])
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--device", default="mps",
                   help="'mps' | 'cuda' | 'cpu'")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grid", action="store_true",
                   help="Use 4×4 224 px tile grid per image (harder)")
    p.add_argument("--hard", action="store_true",
                   help="Restrict gallery to same-slide negatives (harder)")
    p.add_argument("--max_slides", type=int, default=None)
    p.add_argument("--output_json", default=None,
                   help="Save scalar results to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

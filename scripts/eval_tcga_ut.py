#!/usr/bin/env python3
"""
Linear probe evaluation on TCGA-UT (catastrophic-forgetting monitor).

Pipeline
--------
1. Build (or load) a stratified 80/20 slide-level split from the TCGA-UT
   data directory (cached in split_manifest.csv alongside --data_root).
2. Load H0-mini via open_clip's factory (H0-mini-BiomedBERT config):
     --baseline  → pretrained weights, no LoRA
     --checkpoint → open_clip training checkpoint (epoch_*.pt) with LoRA
3. For each slide: load all tiles → extract features → L2-normalise → mean-pool
   → one slide-level feature vector.
4. Train a LogisticRegression on train vectors, evaluate on test vectors.
5. Report accuracy, balanced accuracy, and a per-class classification report.

Usage
-----
  # Baseline (original H0-mini, no LoRA):
  python eval_tcga_ut.py --data_root /path/to/tcga_ut/data --baseline

  # Fine-tuned open_clip checkpoint:
  python eval_tcga_ut.py \\
      --data_root /path/to/tcga_ut/data \\
      --checkpoint logs/<run>/checkpoints/epoch_10.pt

  # Full tiles per slide (slow but accurate):
  python eval_tcga_ut.py ... --max_tiles 0

Adapted from:
  /Users/afiliot/Desktop/libs/constrastive-plism/eval_tcga_probing.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Repository path resolution
# ---------------------------------------------------------------------------
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))

from open_clip.lora_utils import load_visual_from_open_clip_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Pathology normalization (same as training)
# ---------------------------------------------------------------------------
PIXEL_MEAN = [0.707223, 0.578729, 0.703617]
PIXEL_STD  = [0.211883, 0.230117, 0.177517]

# Canonical 31-class ordering for TCGA-UT
TCGA_UT_LABELS: list[str] = [
    "Adrenocortical_carcinoma",
    "Bladder_Urothelial_Carcinoma",
    "Brain_Lower_Grade_Glioma",
    "Breast_invasive_carcinoma",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma",
    "Cholangiocarcinoma",
    "Colon_Rectum_adenocarcinoma",
    "Esophageal_carcinoma",
    "Glioblastoma_multiforme",
    "Head_and_Neck_squamous_cell_carcinoma",
    "Kidney_Chromophobe",
    "Kidney_renal_clear_cell_carcinoma",
    "Kidney_renal_papillary_cell_carcinoma",
    "Liver_hepatocellular_carcinoma",
    "Lung_adenocarcinoma",
    "Lung_squamous_cell_carcinoma",
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
    "Mesothelioma",
    "Ovarian_serous_cystadenocarcinoma",
    "Pancreatic_adenocarcinoma",
    "Pheochromocytoma_and_Paraganglioma",
    "Prostate_adenocarcinoma",
    "Sarcoma",
    "Skin_Cutaneous_Melanoma",
    "Stomach_adenocarcinoma",
    "Testicular_Germ_Cell_Tumors",
    "Thymoma",
    "Thyroid_carcinoma",
    "Uterine_Carcinosarcoma",
    "Uterine_Corpus_Endometrial_Carcinoma",
    "Uveal_Melanoma",
]
LABEL2IDX: dict[str, int] = {l: i for i, l in enumerate(TCGA_UT_LABELS)}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_or_load_manifest(
    root: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Slide-level manifest with stratified train/test split.

    Columns: slide_id | slide_dir | label | label_idx | split

    Cached at ``<root>/../split_manifest.csv``.  Delete to regenerate.
    """
    import pandas as pd

    root = Path(root)
    cache = root.parent / "split_manifest.csv"

    if cache.exists():
        print(f"Loading cached manifest from {cache}")
        return pd.read_csv(cache)

    print(f"Scanning {root} …")
    rows = []
    for slide_dir in sorted(root.iterdir()):
        if not slide_dir.is_dir():
            continue
        jsons = list(slide_dir.glob("*.json"))
        if not jsons:
            continue
        with open(jsons[0]) as f:
            raw = json.load(f)["label"]
        label = raw.replace(" ", "_")
        if label not in LABEL2IDX:
            print(f"  Unknown label '{label}' — skipping {slide_dir.name}")
            continue
        rows.append({
            "slide_id":  slide_dir.name,
            "slide_dir": str(slide_dir),
            "label":     label,
            "label_idx": LABEL2IDX[label],
        })

    df = pd.DataFrame(rows)
    rng = random.Random(seed)
    splits = []
    for _, grp in df.groupby("label"):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        n_test = max(1, round(len(idxs) * test_ratio))
        for i, idx in enumerate(idxs):
            splits.append((idx, "test" if i < n_test else "train"))
    split_s = {idx: sp for idx, sp in splits}
    df["split"] = df.index.map(split_s)
    df.to_csv(cache, index=False)
    print(
        f"Manifest saved to {cache}  "
        f"({len(df)} slides: {(df.split=='train').sum()} train / {(df.split=='test').sum()} test)"
    )
    return df


# ---------------------------------------------------------------------------
# Per-slide tile dataset
# ---------------------------------------------------------------------------

class SlideTileDataset(Dataset):
    """All *.jpg tiles from a single slide directory."""

    def __init__(self, slide_dir: str | Path, transform) -> None:
        self.tile_paths = sorted(Path(slide_dir).glob("*.jpg"))
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.tile_paths[idx]).convert("RGB"))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    manifest_split,
    transform,
    device: torch.device,
    tile_batch_size: int = 32,
    max_tiles: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): mean-pooled slide features and integer labels."""
    backbone.eval()
    use_amp = device.type == "cuda"
    X, y = [], []

    for _, row in tqdm(
        manifest_split.iterrows(), total=len(manifest_split), desc="  slides"
    ):
        tile_ds = SlideTileDataset(row["slide_dir"], transform)
        if len(tile_ds) == 0:
            continue

        if max_tiles and len(tile_ds) > max_tiles:
            indices = torch.randperm(len(tile_ds))[:max_tiles].tolist()
            tile_ds = torch.utils.data.Subset(tile_ds, indices)

        loader = DataLoader(
            tile_ds,
            batch_size=tile_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_amp,
        )

        tile_feats = []
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                feats = backbone(batch)
            feats = F.normalize(feats.float(), dim=-1)
            tile_feats.append(feats.cpu())

        slide_feat = torch.cat(tile_feats, dim=0).mean(dim=0)
        X.append(slide_feat.numpy())
        y.append(int(row["label_idx"]))

    return np.stack(X), np.array(y)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)

    acc     = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    present = sorted(set(y_test))
    report  = classification_report(
        y_test, preds,
        labels=present,
        target_names=[TCGA_UT_LABELS[i] for i in present],
        zero_division=0,
    )
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "report": report}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ------------------------------------------------------------------ model
    # Use open_clip's factory (H0-mini-BiomedBERT JSON config) to load the
    # model, then optionally apply LoRA and load an open_clip checkpoint.
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

    # ---------------------------------------------------------------- splits
    manifest = build_or_load_manifest(
        args.data_root, test_ratio=args.test_ratio, seed=args.seed
    )
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    test_df  = manifest[manifest["split"] == "test"].reset_index(drop=True)

    print(
        f"Slides — train: {len(train_df)}  test: {len(test_df)}  "
        f"classes: {manifest['label'].nunique()}/{len(TCGA_UT_LABELS)}"
    )

    max_tiles = args.max_tiles if args.max_tiles > 0 else None

    # -------------------------------------------------------- feature extraction
    print("Extracting train features …")
    X_train, y_train = extract_features(
        backbone, train_df, transform, device,
        tile_batch_size=args.tile_batch_size, max_tiles=max_tiles,
    )

    print("Extracting test features …")
    X_test, y_test = extract_features(
        backbone, test_df, transform, device,
        tile_batch_size=args.tile_batch_size, max_tiles=max_tiles,
    )

    print(f"Feature shapes: train {X_train.shape}  test {X_test.shape}")

    # ----------------------------------------------------------------- probe
    print("Fitting linear probe …")
    results = run_linear_probe(X_train, y_train, X_test, y_test)

    sep = "=" * 60
    print(
        f"\n{sep}\n"
        f"Accuracy          : {results['accuracy']:.4f}  "
        f"({results['accuracy']*100:.1f}%)\n"
        f"Balanced accuracy : {results['balanced_accuracy']:.4f}  "
        f"({results['balanced_accuracy']*100:.1f}%)\n"
        f"\nPer-class report:\n{results['report']}\n"
        f"{sep}"
    )

    # Save results
    if args.output_json:
        import json as _json
        out = {
            "accuracy": results["accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
            "checkpoint": args.checkpoint,
            "baseline": args.baseline,
            "n_train_slides": len(train_df),
            "n_test_slides":  len(test_df),
        }
        with open(args.output_json, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probe evaluation of H0-mini on TCGA-UT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        default="/Users/afiliot/Desktop/tcga_ut/data",
        help="TCGA-UT data directory (one sub-folder per slide)",
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
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_batch_size", type=int, default=32)
    p.add_argument("--max_tiles", type=int, default=0,
                   help="Max tiles per slide (0 = all)")
    p.add_argument("--output_json", default=None,
                   help="Save scalar results to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

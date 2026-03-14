#!/usr/bin/env python3
"""
LoRA CLIP fine-tuning: H0-mini (vision) + BiomedBERT-large (text) on PathGen-1.6M.

Goal
----
Improve scanner / stain robustness of bioptimus/H0-mini by aligning it with
rich medical text descriptions via contrastive learning (CLIP InfoNCE).
LoRA keeps the number of trainable parameters small and limits catastrophic
forgetting.  An optional knowledge-distillation term further anchors the
LoRA student to the frozen base teacher.

Architecture
------------
  Vision : H0-mini (timm ViT, frozen) + LoRA adapters → vision projection head
  Text   : BiomedBERT-large (HF BERT, frozen) + LoRA adapters → text projection head
  Loss   : CLIP InfoNCE + λ_kd · cosine-distillation (vision only)

Data
----
  --train_data : CSV produced by extract_pathgen_patches.py
    Columns: image_path, caption  (extra columns are ignored)

  --synthetic  : use randomly-generated fake patches + medical captions
    (no real tiles needed; useful for local development / debugging)

Usage
-----
  # Quick local test on Mac (synthetic data, no real tiles needed):
  python train_pathology_clip.py --synthetic --local

  # Single GPU / MPS with real data:
  python train_pathology_clip.py \\
      --train_data /path/to/pathgen_manifest.csv \\
      --output_dir checkpoints/pathgen_clip

  # Multi-GPU (torchrun handles DDP automatically):
  torchrun --nproc_per_node=4 train_pathology_clip.py \\
      --train_data /path/to/pathgen_manifest.csv \\
      --output_dir checkpoints/pathgen_clip \\
      --batch_size 32 --epochs 10

  # Evaluate during training:
  python train_pathology_clip.py ... \\
      --eval_tcga_root  /Users/afiliot/Desktop/tcga_ut/data \\
      --eval_scorpion_root /Users/afiliot/Desktop/scorpion \\
      --eval_interval 500

Requirements
------------
  pip install timm peft transformers torch torchvision Pillow tqdm scikit-learn
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))

from open_clip.lora_utils import (  # noqa: E402
    VisionEncoderWithLoRA,
    TextEncoderWithLoRA,
    FrozenVisionTeacher,
    save_lora_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Pathology-domain normalisation (same for AquaViT, H-optimus-0, H0-mini)
PIXEL_MEAN = [0.707223, 0.578729, 0.703617]
PIXEL_STD  = [0.211883, 0.230117, 0.177517]

# Fake captions for synthetic mode — rotate through these
_FAKE_CAPTIONS = [
    "Hematoxylin and eosin stained section showing invasive ductal carcinoma "
    "with nuclear pleomorphism and mitotic figures.",
    "High-power view of lung adenocarcinoma with acinar growth pattern and "
    "mucin production.",
    "Colon adenocarcinoma with irregular glandular structures infiltrating "
    "the submucosa.",
    "Renal cell carcinoma, clear cell type, with characteristic clear cytoplasm "
    "and delicate vasculature.",
    "Poorly differentiated squamous cell carcinoma with intercellular bridges "
    "and keratin pearl formation.",
    "Hepatocellular carcinoma with trabecular growth pattern and bile production.",
    "Diffuse large B-cell lymphoma with large nuclei, prominent nucleoli, "
    "and high mitotic rate.",
    "Pancreatic ductal adenocarcinoma with desmoplastic stroma and irregular "
    "glandular architecture.",
    "Melanoma with epithelioid cells, intranuclear inclusions, and melanin "
    "pigment deposits.",
    "Glioblastoma multiforme showing pseudopalisading necrosis, endothelial "
    "proliferation, and nuclear atypia.",
    "Prostatic adenocarcinoma with small acinar pattern, loss of basal layer, "
    "and perineural invasion.",
    "Cervical squamous cell carcinoma with keratinisation and stromal invasion.",
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PathGenDataset(Dataset):
    """Image–caption pairs from the PathGen-1.6M manifest CSV."""

    def __init__(
        self,
        csv_path: str,
        transform,
        max_samples: Optional[int] = None,
    ) -> None:
        self.transform = transform
        self.rows: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
                if max_samples and len(self.rows) >= max_samples:
                    break
        log.info(f"PathGen dataset: {len(self.rows):,} image–caption pairs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.rows[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            return self.transform(img), row["caption"]
        except Exception:
            rand_idx = random.randint(0, len(self.rows) - 1)
            return self.__getitem__(rand_idx)


class SyntheticPathGenDataset(Dataset):
    """Randomly-generated fake patches + cycling medical captions.

    Produces tensors directly — no disk I/O — so it works anywhere.
    Patches are Gaussian noise with pathology-plausible colour statistics
    (pinkish H&E tones, then normalised with pathology norms).
    """

    def __init__(
        self,
        size: int = 2048,
        input_size: int = 224,
        seed: int = 42,
    ) -> None:
        self.size       = size
        self.input_size = input_size
        self._rng       = np.random.default_rng(seed)
        # Pre-generate all images as uint8 arrays for speed
        log.info(f"Synthetic dataset: {size} fake image–caption pairs")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        rng = np.random.default_rng(idx)
        # H&E-like colour: pinkish (high R, medium G, high B)
        base_rgb  = rng.integers(180, 230, size=(3,), dtype=np.uint8)
        noise     = rng.normal(0, 20, size=(self.input_size, self.input_size, 3))
        img_arr   = np.clip(
            base_rgb[None, None, :] + noise, 0, 255
        ).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="RGB")

        # Normalise with pathology statistics
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
        ])(img)

        caption = _FAKE_CAPTIONS[idx % len(_FAKE_CAPTIONS)]
        return tensor, caption


def build_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

class PathologyCLIP(nn.Module):
    def __init__(
        self,
        vision_backbone: VisionEncoderWithLoRA,
        text_encoder: TextEncoderWithLoRA,
        embed_dim: int = 512,
        init_logit_scale: float = math.log(1 / 0.07),
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.text_encoder    = text_encoder
        self.vision_proj = nn.Linear(vision_backbone.embed_dim, embed_dim, bias=False)
        self.text_proj   = nn.Linear(text_encoder.embed_dim,   embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        nn.init.normal_(self.vision_proj.weight, std=vision_backbone.embed_dim ** -0.5)
        nn.init.normal_(self.text_proj.weight,   std=text_encoder.embed_dim   ** -0.5)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision_proj(self.vision_backbone(images)), dim=-1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        return F.normalize(self.text_proj(self.text_encoder(input_ids, attention_mask)), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        img_feats  = self.encode_image(images)
        text_feats = self.encode_text(input_ids, attention_mask)
        return img_feats, text_feats, self.logit_scale.exp().clamp(max=100)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def clip_loss(image_features, text_features, logit_scale) -> torch.Tensor:
    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(len(image_features), device=image_features.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def distillation_loss(student_feats, teacher_feats) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student_feats, teacher_feats, dim=-1)).mean()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def setup_distributed() -> tuple[int, int, bool]:
    """Initialise DDP if launched with torchrun; fall back to single-process."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, False   # non-distributed

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size, True


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, warmup_steps, total_steps):
    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Mid-training evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_tcga_eval(backbone, data_root, device, tile_batch_size=32, max_tiles=50):
    """Quick linear probe on TCGA-UT."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    import json as _json

    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])
    LABEL2IDX: dict[str, int] = {}
    rows = []
    for slide_dir in sorted(Path(data_root).iterdir()):
        if not slide_dir.is_dir():
            continue
        jsons = list(slide_dir.glob("*.json"))
        if not jsons:
            continue
        with open(jsons[0]) as f:
            label = _json.load(f)["label"].replace(" ", "_")
        if label not in LABEL2IDX:
            LABEL2IDX[label] = len(LABEL2IDX)
        rows.append({"slide_dir": slide_dir, "label_idx": LABEL2IDX[label]})
    if not rows:
        return {}

    rng = random.Random(42)
    by_label: dict[int, list] = {}
    for row in rows:
        by_label.setdefault(row["label_idx"], []).append(row)
    train_rows, test_rows = [], []
    for lbl_rows in by_label.values():
        rng.shuffle(lbl_rows)
        n_test = max(1, round(len(lbl_rows) * 0.2))
        test_rows.extend(lbl_rows[:n_test])
        train_rows.extend(lbl_rows[n_test:])

    backbone.eval()

    def _extract(split_rows):
        X, y = [], []
        for row in split_rows:
            tile_paths = sorted(Path(row["slide_dir"]).glob("*.jpg"))
            if not tile_paths:
                continue
            if max_tiles and len(tile_paths) > max_tiles:
                tile_paths = random.sample(tile_paths, max_tiles)
            tile_feats = []
            for i in range(0, len(tile_paths), tile_batch_size):
                imgs = torch.stack([
                    transform(Image.open(p).convert("RGB"))
                    for p in tile_paths[i: i + tile_batch_size]
                ]).to(device)
                tile_feats.append(F.normalize(backbone(imgs).float(), dim=-1).cpu())
            slide_feat = torch.cat(tile_feats).mean(0).numpy()
            X.append(slide_feat)
            y.append(row["label_idx"])
        return np.array(X), np.array(y)

    X_tr, y_tr = _extract(train_rows)
    X_te, y_te = _extract(test_rows)
    if not len(X_tr) or not len(X_te):
        return {}
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    preds = clf.predict(sc.transform(X_te))
    return {
        "tcga_acc":     float(accuracy_score(y_te, preds)),
        "tcga_bal_acc": float(balanced_accuracy_score(y_te, preds)),
    }


@torch.no_grad()
def _run_scorpion_eval(backbone, data_root, device, batch_size=16, max_slides=None):
    """Quick SCORPION retrieval (R@1, mAP)."""
    SCANNERS = ["AT2", "DP200", "GT450", "P1000", "Philips"]
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
    ])
    items = []
    slide_dirs = sorted(
        [p for p in Path(data_root).iterdir() if p.is_dir() and p.name.startswith("slide_")],
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
            for scanner in SCANNERS:
                p = sample_dir / f"{scanner}.jpg"
                if p.exists():
                    items.append({"path": p, "group_id": s_idx * 10_000 + sa_idx})
    if not items:
        return {}

    backbone.eval()
    all_feats, all_groups = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        imgs = torch.stack([
            transform(Image.open(it["path"]).convert("RGB")) for it in batch
        ]).to(device)
        feats = F.normalize(backbone(imgs).float(), dim=-1)
        all_feats.append(feats.cpu())
        all_groups.extend(it["group_id"] for it in batch)

    features  = torch.cat(all_feats).numpy()
    group_arr = np.array(all_groups)
    N = len(features)
    sim = (torch.from_numpy(features) @ torch.from_numpy(features).T).numpy()
    np.fill_diagonal(sim, -2.0)
    ranked     = np.argsort(-sim, axis=1)
    pos_ranked = group_arr[ranked] == group_arr[:, None]
    self_mask  = ranked == np.arange(N)[:, None]
    pos_ranked &= ~self_mask
    r1       = pos_ranked[:, 0].astype(float).mean()
    cumhits  = pos_ranked.cumsum(axis=1).astype(float)
    n_pos    = pos_ranked.sum(axis=1)
    valid    = n_pos > 0
    ranks    = np.arange(1, N + 1, dtype=float)
    ap       = np.where(valid, (cumhits / ranks * pos_ranked).sum(axis=1) / np.maximum(n_pos, 1), 0.0)
    return {"scorpion_r1": float(r1), "scorpion_map": float(ap.mean())}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------ distributed
    local_rank, world_size, distributed = setup_distributed()
    main_proc = is_main_process()

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if main_proc:
        log.info(f"Device: {device}  |  world_size={world_size}  |  distributed={distributed}")

    torch.manual_seed(args.seed + local_rank)
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # ----------------------------------------------------------------- models
    if main_proc:
        log.info(f"Loading vision backbone: {args.vision_model}")
    vision_backbone = VisionEncoderWithLoRA(
        model_name=args.vision_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.vision_target_modules,
        model_local_dir=args.vision_model_local_dir,
    )
    if main_proc:
        vision_backbone.print_trainable_summary()
        log.info(f"Loading text encoder: {args.text_model}")
    text_encoder = TextEncoderWithLoRA(
        model_name_or_path=args.text_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.text_target_modules,
        max_length=args.max_text_length,
    )
    if main_proc:
        text_encoder.print_trainable_summary()

    model = PathologyCLIP(vision_backbone, text_encoder, args.embed_dim).to(device)

    # Frozen teacher for knowledge distillation
    teacher: Optional[FrozenVisionTeacher] = None
    if args.distill_lambda > 0.0:
        if main_proc:
            log.info("Building frozen teacher for knowledge distillation …")
        teacher = FrozenVisionTeacher(
            model_name=args.vision_model,
            model_local_dir=args.vision_model_local_dir,
        ).to(device)

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        # Unwrap for convenience
        raw_model: PathologyCLIP = model.module  # type: ignore[assignment]
    else:
        raw_model = model

    # ------------------------------------------------------------------- data
    if args.synthetic:
        if main_proc:
            log.info(f"Using synthetic dataset ({args.synthetic_size} fake samples)")
        dataset = SyntheticPathGenDataset(
            size=args.synthetic_size,
            input_size=args.input_size,
            seed=args.seed,
        )
    else:
        train_transform = build_train_transform(args.input_size)
        dataset = PathGenDataset(
            csv_path=args.train_data,
            transform=train_transform,
            max_samples=args.max_train_samples,
        )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # -------------------------------------------------------------- optimiser
    trainable_params = (
        list(raw_model.vision_backbone.trainable_parameters())
        + list(raw_model.text_encoder.trainable_parameters())
        + list(raw_model.vision_proj.parameters())
        + list(raw_model.text_proj.parameters())
        + [raw_model.logit_scale]
    )
    if main_proc:
        log.info(
            f"Trainable params: "
            f"{sum(p.numel() for p in trainable_params) / 1e6:.2f} M"
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    steps_per_epoch = len(loader) // args.grad_accum_steps
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * args.warmup_epochs
    scheduler = cosine_lr(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --------------------------------------------------------------- output
    output_dir = Path(args.output_dir)
    if main_proc:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(output_dir / "config.json", "w") as f:
            _json.dump(vars(args), f, indent=2)

    # ----------------------------------------------------------------- train
    global_step      = 0
    best_scorpion_r1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if distributed and sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not main_proc, leave=False)
        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            # Tokenise (text_encoder not wrapped in DDP — lives on all ranks identically)
            encoded = raw_model.text_encoder.tokenizer(
                list(captions),
                padding=True,
                truncation=True,
                max_length=args.max_text_length,
                return_tensors="pt",
            )
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                img_feats, text_feats, logit_scale = model(images, input_ids, attention_mask)
                loss = clip_loss(img_feats, text_feats, logit_scale)

                if teacher is not None and args.distill_lambda > 0.0:
                    with torch.no_grad():
                        teacher_feats = teacher(images)
                    student_raw = raw_model.vision_backbone(images)
                    loss = loss + args.distill_lambda * distillation_loss(student_raw, teacher_feats)

            scaler.scale(loss / args.grad_accum_steps).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if main_proc and global_step % args.log_interval == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")
                    log.info(
                        f"  step {global_step:6d}  loss={loss.item():.4f}  "
                        f"scale={logit_scale.item():.2f}  lr={lr_now:.2e}"
                    )

                # ----------------------------------------- mid-training eval
                if (
                    main_proc
                    and args.eval_interval > 0
                    and global_step % args.eval_interval == 0
                ):
                    model.eval()
                    eval_results: dict[str, float] = {}
                    if args.eval_tcga_root:
                        eval_results.update(_run_tcga_eval(
                            raw_model.vision_backbone, args.eval_tcga_root, device,
                            tile_batch_size=args.eval_batch_size,
                            max_tiles=args.eval_max_tiles,
                        ))
                    if args.eval_scorpion_root:
                        eval_results.update(_run_scorpion_eval(
                            raw_model.vision_backbone, args.eval_scorpion_root, device,
                            batch_size=args.eval_batch_size,
                            max_slides=args.eval_max_slides,
                        ))
                    if eval_results and main_proc:
                        parts = "  ".join(f"{k}={v:.4f}" for k, v in eval_results.items())
                        log.info(f"  [eval step {global_step}]  {parts}")

                    r1 = eval_results.get("scorpion_r1", -1.0)
                    if r1 > best_scorpion_r1:
                        best_scorpion_r1 = r1
                        save_lora_checkpoint(
                            path=output_dir / "best_checkpoint.pt",
                            vision_backbone=raw_model.vision_backbone,
                            text_encoder=raw_model.text_encoder,
                            vision_proj=raw_model.vision_proj,
                            text_proj=raw_model.text_proj,
                            logit_scale=raw_model.logit_scale.item(),
                            epoch=epoch, step=global_step, cfg=vars(args),
                        )
                        log.info(f"  ✓ Best SCORPION R@1={r1:.4f}")
                    model.train()

        if main_proc:
            elapsed = time.time() - t0
            n_b     = len(loader)
            log.info(
                f"Epoch {epoch}/{args.epochs}  "
                f"avg_loss={epoch_loss/n_b:.4f}  "
                f"time={elapsed:.0f}s"
            )
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_lora_checkpoint(
                path=ckpt_path,
                vision_backbone=raw_model.vision_backbone,
                text_encoder=raw_model.text_encoder,
                vision_proj=raw_model.vision_proj,
                text_proj=raw_model.text_proj,
                logit_scale=raw_model.logit_scale.item(),
                epoch=epoch, step=global_step, cfg=vars(args),
            )
            log.info(f"Checkpoint → {ckpt_path}")
            if args.keep_last_n > 0:
                all_ckpts = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
                for old in all_ckpts[: -args.keep_last_n]:
                    old.unlink()

    cleanup_distributed(distributed)
    if main_proc:
        log.info(f"\nTraining complete. Checkpoints in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA CLIP fine-tuning: H0-mini + BiomedBERT-large on PathGen-1.6M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- shortcut: local Mac quick-test preset ----
    p.add_argument(
        "--local", action="store_true",
        help="Mac-friendly preset: synthetic data, tiny LoRA, small batch, MPS/CPU",
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--train_data", default=None,
                   help="Path to pathgen_manifest.csv")
    g.add_argument("--synthetic", action="store_true",
                   help="Use synthetic (random) patches instead of real tiles")
    g.add_argument("--synthetic_size", type=int, default=2048,
                   help="Number of fake samples in synthetic mode")
    g.add_argument("--max_train_samples", type=int, default=None)
    g.add_argument("--input_size", type=int, default=224)
    g.add_argument("--max_text_length", type=int, default=256)

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--vision_model", default="hf-hub:bioptimus/H0-mini")
    g.add_argument("--vision_model_local_dir", default=None)
    g.add_argument("--text_model",
                   default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext")
    g.add_argument("--embed_dim", type=int, default=512)

    # LoRA
    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_r", type=int, default=8)
    g.add_argument("--lora_alpha", type=int, default=16)
    g.add_argument("--lora_dropout", type=float, default=0.05)
    g.add_argument("--vision_target_modules", nargs="+", default=["qkv"])
    g.add_argument("--text_target_modules", nargs="+",
                   default=["query", "key", "value"])

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--warmup_epochs", type=int, default=1)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--grad_accum_steps", type=int, default=1)
    g.add_argument("--distill_lambda", type=float, default=0.1)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int, default=42)

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="checkpoints/pathgen_clip")
    g.add_argument("--log_interval", type=int, default=50)
    g.add_argument("--keep_last_n", type=int, default=3)

    # Evaluation during training
    g = p.add_argument_group("Mid-training evaluation")
    g.add_argument("--eval_interval", type=int, default=0)
    g.add_argument("--eval_tcga_root",
                   default="/Users/afiliot/Desktop/tcga_ut/data")
    g.add_argument("--eval_scorpion_root",
                   default="/Users/afiliot/Desktop/scorpion")
    g.add_argument("--eval_batch_size", type=int, default=32)
    g.add_argument("--eval_max_tiles", type=int, default=50)
    g.add_argument("--eval_max_slides", type=int, default=None)

    args = p.parse_args()

    # Apply --local preset AFTER parsing so explicit flags can still override
    if args.local:
        args.synthetic       = args.synthetic or True
        args.synthetic_size  = args.synthetic_size if args.synthetic_size != 2048 else 512
        args.batch_size      = min(args.batch_size, 8)
        args.lora_r          = min(args.lora_r, 2)
        args.lora_alpha      = min(args.lora_alpha, 4)
        args.distill_lambda  = 0.0        # skip teacher to save memory
        args.epochs          = min(args.epochs, 3)
        args.num_workers     = 0          # avoid fork issues on macOS
        args.log_interval    = 5
        args.eval_interval   = 0
        args.output_dir      = args.output_dir or "checkpoints/local_test"

    if not args.synthetic and args.train_data is None:
        p.error("Provide --train_data or use --synthetic (or --local)")

    return args


if __name__ == "__main__":
    train(parse_args())

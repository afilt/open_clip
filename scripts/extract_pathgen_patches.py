#!/usr/bin/env python3
"""
Extract histology patches from TCGA whole-slide images for PathGen-1.6M.

Pipeline
--------
1. Read PathGen-1.6M.json → group entries by file_id (one WSI per group).
2. For each WSI: find the file in --slides_dir, open with OpenSlide, extract
   672×672 px patches at the (x, y) coordinates from the metadata
   (level 0, 0.5 µm/px).
3. Save patches as PNG and append rows to a manifest CSV ready for
   ``--train-data`` in the CLIP fine-tuning script.

Slide directory
---------------
All WSI files (.svs / .tiff / .tif / .ndpi / .scn) must live flat in
--slides_dir (no sub-folders needed). Default: /lustre/fsmisc/dataset/TCGA_WSI

Matching order for each WSI:
  1. File stem == file_id  (e.g. <uuid>.svs)
  2. File stem == wsi_id   (e.g. TCGA-XX-XXXX-….svs)
  3. Either ID appears as a substring of the stem

Requirements
------------
    pip install openslide-python tqdm

Usage
-----
    python extract_pathgen_patches.py \\
        --metadata   /Users/afiliot/Desktop/pathgen-1.6M/PathGen-1.6M.json \\
        --slides_dir /lustre/fsmisc/dataset/TCGA_WSI \\
        --output_dir /path/to/output

    python extract_pathgen_patches.py ... --max_slides 5   # test on 5 slides
    python extract_pathgen_patches.py ... --resume         # skip done slides

Output layout
-------------
    <output_dir>/
      patches/
        <wsi_id>/
          <x>_<y>.png      ← 672×672 RGB patch
      pathgen_manifest.csv ← image_path, caption, wsi_id, file_id, x, y
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PATCH_SIZE = 672  # pixels at WSI level 0

# Flat folder where all TCGA WSIs will be stored on the cluster
DEFAULT_SLIDES_DIR = "/lustre/fsmisc/dataset/TCGA_WSI"

WSI_EXTENSIONS = {".svs", ".tiff", ".tif", ".ndpi", ".scn"}


# ---------------------------------------------------------------------------
# Local slide index (mode A)
# ---------------------------------------------------------------------------

def build_slide_index(slides_dir: str) -> dict[str, Path]:
    """Scan a flat directory and return a mapping from identifier → WSI path.

    The index is keyed by both the file stem (e.g. the GDC file UUID when
    files are named ``<file_id>.svs``) and any prefix that looks like a
    TCGA barcode (``TCGA-XX-XXXX-…``).  Matching is done in
    :func:`find_local_slide`.
    """
    index: dict[str, Path] = {}
    root = Path(slides_dir)
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in WSI_EXTENSIONS:
            index[p.stem.lower()] = p
    log.info(f"Slide index: {len(index):,} files found in {slides_dir}")
    return index


def find_local_slide(
    file_id: str,
    wsi_id: str,
    index: dict[str, Path],
) -> Optional[Path]:
    """Return the local WSI path for a given (file_id, wsi_id) pair.

    Matching order:
    1. Exact stem match on file_id   (files named ``<uuid>.svs``)
    2. Exact stem match on wsi_id    (files named ``<barcode>.svs``)
    3. Any file whose stem contains wsi_id as a substring
    """
    key = file_id.lower()
    if key in index:
        return index[key]
    key2 = wsi_id.lower()
    if key2 in index:
        return index[key2]
    for stem, path in index.items():
        if key2 in stem or key in stem:
            return path
    return None


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(json_path: str) -> List[Dict[str, Any]]:
    log.info(f"Loading metadata from {json_path} …")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"  → {len(data):,} entries loaded")
    return data


def group_by_file_id(
    entries: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect entries sharing the same GDC file UUID (= same WSI download)."""
    groups: Dict[str, list] = {}
    for e in entries:
        groups.setdefault(e["file_id"], []).append(e)
    return groups


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches_from_slide(
    wsi_path: Path,
    entries: List[Dict[str, Any]],
    patches_dir: Path,
    patch_size: int = PATCH_SIZE,
) -> List[Dict[str, str]]:
    """Open `wsi_path` with OpenSlide and extract all patches in `entries`.

    Returns a list of manifest rows (dicts with image_path, caption, …).
    """
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "openslide-python is required: pip install openslide-python\n"
            "You also need the OpenSlide C library: "
            "https://openslide.org/download/"
        ) from exc

    results: List[Dict[str, str]] = []

    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except openslide.OpenSlideError as exc:
        log.error(f"Cannot open {wsi_path}: {exc}")
        return results

    wsi_id = entries[0]["wsi_id"]
    out_dir = patches_dir / wsi_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        x = int(entry["position"][0])
        y = int(entry["position"][1])
        patch_path = out_dir / f"{x}_{y}.png"

        if patch_path.exists():
            # Already extracted (resume mode)
            results.append(_make_row(patch_path, entry, wsi_id))
            continue

        try:
            region = slide.read_region((x, y), 0, (patch_size, patch_size))
            region = region.convert("RGB")
            region.save(str(patch_path), "PNG")
            results.append(_make_row(patch_path, entry, wsi_id))
        except Exception as exc:
            log.warning(f"Failed to extract ({x},{y}) from {wsi_id}: {exc}")

    slide.close()
    return results


def _make_row(
    patch_path: Path, entry: Dict[str, Any], wsi_id: str
) -> Dict[str, str]:
    return {
        "image_path": str(patch_path),
        "caption":    entry["caption"],
        "wsi_id":     wsi_id,
        "file_id":    entry["file_id"],
        "x":          str(int(entry["position"][0])),
        "y":          str(int(entry["position"][1])),
    }


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

MANIFEST_FIELDNAMES = ["image_path", "caption", "wsi_id", "file_id", "x", "y"]


def load_existing_manifest(manifest_path: Path) -> tuple[List[Dict[str, str]], set]:
    """Load an existing manifest and return (rows, set_of_completed_file_ids)."""
    rows: List[Dict[str, str]] = []
    done: set = set()
    if not manifest_path.exists():
        return rows, done
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            done.add(row["file_id"])
    log.info(f"Resume: {len(done)} slides already done, {len(rows):,} patches")
    return rows, done


def append_manifest(manifest_path: Path, rows: List[Dict[str, str]]) -> None:
    """Append `rows` to the manifest CSV (create with header if needed)."""
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-slide worker
# ---------------------------------------------------------------------------

def process_slide(
    file_id: str,
    entries: List[Dict[str, Any]],
    patches_dir: Path,
    manifest_path: Path,
    slide_index: dict,
    patch_size: int,
) -> int:
    """Locate one WSI in the local slide index, extract patches, update manifest.

    Returns the number of successfully extracted patches.
    """
    wsi_id = entries[0]["wsi_id"]
    log.info(f"▶  {wsi_id}  ({len(entries)} patches, file_id={file_id})")

    wsi_path = find_local_slide(file_id, wsi_id, slide_index)
    if wsi_path is None:
        log.warning(f"Skipping {wsi_id}: not found in slides_dir")
        return 0

    rows = extract_patches_from_slide(wsi_path, entries, patches_dir, patch_size)

    if rows:
        append_manifest(manifest_path, rows)
        log.info(f"   ✓  {len(rows)}/{len(entries)} patches saved for {wsi_id}")

    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract PathGen-1.6M patches from TCGA whole-slide images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metadata",
        default="/Users/afiliot/Desktop/pathgen-1.6M/PathGen-1.6M.json",
        help="Path to PathGen-1.6M.json",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory (patches/ and pathgen_manifest.csv go here)",
    )

    p.add_argument(
        "--slides_dir",
        default=DEFAULT_SLIDES_DIR,
        help=(
            "Flat directory containing all WSI files (.svs/.tiff/…). "
            f"Default: {DEFAULT_SLIDES_DIR}"
        ),
    )

    p.add_argument(
        "--max_slides",
        type=int,
        default=None,
        help="Stop after this many slides (useful for testing)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip slides already present in pathgen_manifest.csv",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        default=PATCH_SIZE,
        help="Patch edge length in pixels (level 0, 0.5 µm/px → 672 px = 336 µm)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Process slides in random order (for diverse early checkpoints)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir    = Path(args.output_dir)
    patches_dir   = output_dir / "patches"
    manifest_path = output_dir / "pathgen_manifest.csv"

    patches_dir.mkdir(parents=True, exist_ok=True)

    slides_dir = Path(args.slides_dir)
    if not slides_dir.is_dir():
        raise SystemExit(f"--slides_dir '{slides_dir}' does not exist or is not a directory.")
    log.info(f"Using local slides from {slides_dir}")
    slide_index = build_slide_index(str(slides_dir))

    entries = load_metadata(args.metadata)
    grouped = group_by_file_id(entries)
    all_file_ids = list(grouped.keys())
    log.info(f"Unique WSIs: {len(all_file_ids):,}")

    # Resume
    existing_rows, done_ids = (
        load_existing_manifest(manifest_path) if args.resume else ([], set())
    )

    remaining = [fid for fid in all_file_ids if fid not in done_ids]

    if args.shuffle:
        import random
        random.shuffle(remaining)

    if args.max_slides is not None:
        remaining = remaining[: args.max_slides]

    log.info(
        f"Slides to process: {len(remaining):,} "
        f"(skipped {len(done_ids):,} already done)"
    )

    total_patches = len(existing_rows)
    for file_id in tqdm(remaining, desc="Slides", unit="slide"):
        n = process_slide(
            file_id=file_id,
            entries=grouped[file_id],
            patches_dir=patches_dir,
            manifest_path=manifest_path,
            slide_index=slide_index,
            patch_size=args.patch_size,
        )
        total_patches += n

    log.info(
        f"\nDone. Total patches in manifest: {total_patches:,}\n"
        f"Manifest: {manifest_path}\n"
        f"Patches:  {patches_dir}"
    )


if __name__ == "__main__":
    main()

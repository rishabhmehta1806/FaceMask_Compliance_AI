#!/usr/bin/env python3
"""Safely import the Kaggle FMD_DATASET into MaskGuard's class folders.

Expected source layout:
  FMD_DATASET/incorrect_mask/mc       -> below_chin
  FMD_DATASET/incorrect_mask/mmc      -> nose_exposed
  FMD_DATASET/with_mask/{simple,complex} -> correctly_worn
  FMD_DATASET/without_mask/{simple,complex} -> no_mask

The source dataset is never modified. By default this script creates hard links
(on the same filesystem) and falls back to copying. Use --copy to force copies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED_SOURCE = {
    "incorrect_mask/mc": "below_chin",
    "incorrect_mask/mmc": "nose_exposed",
    "with_mask/simple": "correctly_worn",
    "with_mask/complex": "correctly_worn",
    "without_mask/simple": "no_mask",
    "without_mask/complex": "no_mask",
}
ALL_CLASSES = ("correctly_worn", "nose_exposed", "mouth_exposed", "below_chin", "no_mask")


def image_is_readable(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        # Reopen after verify and reject implausibly tiny files/dimensions.
        with Image.open(path) as image:
            width, height = image.size
            if width < 32 or height < 32:
                return False, f"too_small:{width}x{height}"
        return True, "ok"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"unreadable:{type(exc).__name__}"


def make_target_name(source: Path, relative: Path) -> str:
    # Keep the original stem so the project's group-aware split can group
    # same-named source examples across classes; hash prevents name collisions.
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{source.stem}_{digest}{source.suffix.lower()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import FMD images into data/input.")
    parser.add_argument("--source", default="data/raw/sfmd/FMD_DATASET", help="Extracted FMD_DATASET folder")
    parser.add_argument("--destination", default="data/input", help="Five-class input folder")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking them")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report, without importing")
    parser.add_argument("--manifest", default="reports/fmd_import_manifest.csv", help="CSV manifest path")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    destination_root = Path(args.destination).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not source_root.is_dir():
        parser.error(f"Source directory does not exist: {source_root}")

    for class_name in ALL_CLASSES:
        (destination_root / class_name).mkdir(parents=True, exist_ok=True)

    summary: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    seen_source_paths = 0

    for relative_dir, label in EXPECTED_SOURCE.items():
        source_dir = source_root / relative_dir
        if not source_dir.is_dir():
            print(f"WARNING: expected source folder missing: {source_dir}")
            summary[f"missing_folder:{relative_dir}"] += 1
            continue

        files = sorted(
            p for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        print(f"{relative_dir:32s} -> {label:15s}: {len(files):5d} image file(s)")

        for path in files:
            seen_source_paths += 1
            readable, status = image_is_readable(path)
            target_dir = destination_root / label
            relative = path.relative_to(source_root)
            target = target_dir / make_target_name(path, relative)

            if not readable:
                summary[f"bad:{label}"] += 1
                rows.append({"source": str(path), "label": label, "target": "", "status": status})
                continue

            if args.dry_run:
                summary[f"valid:{label}"] += 1
                rows.append({"source": str(path), "label": label, "target": str(target), "status": "dry_run_valid"})
                continue

            if target.exists():
                summary[f"already_present:{label}"] += 1
                rows.append({"source": str(path), "label": label, "target": str(target), "status": "already_present"})
                continue

            try:
                if args.copy:
                    shutil.copy2(path, target)
                    method = "copied"
                else:
                    try:
                        os.link(path, target)
                        method = "hardlinked"
                    except OSError:
                        shutil.copy2(path, target)
                        method = "copied_fallback"
                summary[f"imported:{label}"] += 1
                rows.append({"source": str(path), "label": label, "target": str(target), "status": method})
            except OSError as exc:
                summary[f"failed:{label}"] += 1
                rows.append({"source": str(path), "label": label, "target": str(target), "status": f"failed:{type(exc).__name__}"})

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "label", "target", "status"])
            writer.writeheader()
            writer.writerows(rows)

    print("\nSUMMARY")
    print(f"Source image files scanned: {seen_source_paths}")
    for label in ALL_CLASSES:
        if args.dry_run:
            good = summary[f"valid:{label}"]
            bad = summary[f"bad:{label}"]
            print(f"  {label:15s}: {good:5d} readable, {bad:4d} unreadable")
        else:
            print(
                f"  {label:15s}: {summary[f'imported:{label}']:5d} imported, "
                f"{summary[f'already_present:{label}']:4d} already present, "
                f"{summary[f'bad:{label}']:4d} unreadable, "
                f"{summary[f'failed:{label}']:4d} failed"
            )
    print(f"  mouth_exposed: 0 imported (this dataset has no separately labelled mouth-exposed class)")
    if args.dry_run:
        print("\nDry run only: no images were imported and no manifest was written.")
    else:
        print(f"\nManifest written to: {manifest_path}")
        print("Original source images were left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

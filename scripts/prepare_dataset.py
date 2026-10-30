from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED = ["correctly_worn", "nose_exposed", "below_chin", "no_mask"]

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test folders with group-aware splitting.")
    parser.add_argument("--input", default="data/input")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-missing-classes", action="store_true", help="For quick experiments only; not recommended for final results.")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    class_files = {}
    for label in EXPECTED:
        folder = input_dir / label
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS] if folder.exists() else []
        class_files[label] = files
    missing = [name for name, files in class_files.items() if len(files) < 3]
    if missing and not args.allow_missing_classes:
        raise RuntimeError(
            "At least 3 images per class are required for a smoke test; for a credible report collect hundreds+ per class. "
            f"Missing/too-small classes: {missing}. Expected folders under {input_dir}."
        )
    class_files = {k: v for k, v in class_files.items() if v}
    if not class_files:
        raise RuntimeError("No labelled images were found.")

    # Keep identical source stems together across classes (important for MaskedFace-Net/FFHQ variants).
    records = []
    for label, files in class_files.items():
        for path in files:
            group = path.stem.rsplit("_", 1)[0] if len(path.stem.rsplit("_", 1)) == 2 and len(path.stem.rsplit("_", 1)[1]) == 8 else path.stem
            records.append((path, label, group))

    # Retry group-shuffle splits until all available classes occur in all partitions.
    from sklearn.model_selection import GroupShuffleSplit
    available_labels = set(class_files)
    selected = None
    for attempt in range(300):
        seed = args.seed + attempt
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(records, [r[1] for r in records], [r[2] for r in records]))
        trainval = [records[i] for i in trainval_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 1000)
        train_idx, val_idx = next(gss_val.split(trainval, [r[1] for r in trainval], [r[2] for r in trainval]))
        train = [trainval[i] for i in train_idx]
        val = [trainval[i] for i in val_idx]
        test = [records[i] for i in test_idx]
        if all({r[1] for r in split} == available_labels for split in (train, val, test)):
            selected = {"train": train, "val": val, "test": test}
            break
    if selected is None:
        raise RuntimeError(
            "Could not create a group-aware split containing every class in train/val/test. "
            "Add more independent source images per class or use unique source filenames."
        )

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists. Pass --force to replace it.")
        shutil.rmtree(output_dir)
    for split_name, split_records in selected.items():
        counts = defaultdict(int)
        for src, label, _ in split_records:
            out_dir = output_dir / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            # File names are made unique within class/split.
            digest = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:10]
            target = out_dir / f"{src.stem}_{digest}{src.suffix.lower()}"
            shutil.copy2(src, target)
            counts[label] += 1
        print(f"{split_name}: {dict(counts)} ({len(split_records)} images)")

    print(f"Dataset prepared at: {output_dir.resolve()}")
    print("Split is source-stem-group-aware; keep original stems when importing data to reduce identity leakage.")


if __name__ == "__main__":
    main()

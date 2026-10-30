from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def infer_label(path: Path) -> str | None:
    parts = [p.lower().replace("-", "_").replace(" ", "_") for p in path.parts]
    joined = "/".join(parts)
    name = path.stem.lower()
    # MaskedFace-Net labels: CMFD (correct), IMFD2 (nose uncovered), IMFD3 (mouth uncovered).
    if "imfd1" in joined or "imfd1" in name:
        return None  # uncovered chin is NOT the same as mask-below-chin.
    if "imfd2" in joined or "imfd2" in name or "nose_uncovered" in joined or "uncovered_nose" in joined:
        return "nose_exposed"
    if "imfd3" in joined or "imfd3" in name or "mouth_uncovered" in joined or "uncovered_mouth" in joined:
        return "mouth_exposed"
    if "cmfd" in joined or "correctly_masked" in joined or "correct_mask" in joined:
        return "correctly_worn"
    return None


def main():
    parser = argparse.ArgumentParser(description="Import supported MaskedFace-Net subsets into data/input.")
    parser.add_argument("--source", required=True, help="Extracted MaskedFace-Net root folder")
    parser.add_argument("--destination", default="data/input")
    parser.add_argument("--copy", action="store_true", help="Copy instead of creating hard links")
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    if not source.exists():
        raise FileNotFoundError(source)
    counts = {"correctly_worn": 0, "nose_exposed": 0, "mouth_exposed": 0}
    skipped = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = infer_label(path)
        if label is None:
            skipped += 1
            continue
        out_dir = destination / label
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
        target = out_dir / f"{path.stem}_{digest}{path.suffix.lower()}"
        if args.copy:
            shutil.copy2(path, target)
        else:
            try:
                target.hardlink_to(path)
            except OSError:
                shutil.copy2(path, target)
        counts[label] += 1
    print("Imported samples:")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    print(f"  skipped/unmapped: {skipped}")
    print("NOTE: IMFD1 (chin uncovered) was skipped because it is not equivalent to the requested below_chin class.")
    print("Add real examples in data/input/below_chin and data/input/no_mask before training the five-class model.")


if __name__ == "__main__":
    main()

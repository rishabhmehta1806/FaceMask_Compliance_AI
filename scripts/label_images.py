from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import cv2

LABELS = {
    ord("1"): "correctly_worn",
    ord("2"): "nose_exposed",
    ord("3"): "mouth_exposed",
    ord("4"): "below_chin",
    ord("5"): "no_mask",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Simple human-in-the-loop image labeller.")
    parser.add_argument("--source", required=True, help="Folder containing unlabelled face crops")
    parser.add_argument("--destination", default="data/input")
    args = parser.parse_args()
    source, dest = Path(args.source), Path(args.destination)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    print("Keys: 1 correct | 2 nose exposed | 3 mouth exposed | 4 below chin | 5 no mask | s skip | q quit")
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        preview = cv2.resize(image, (640, 640)) if image.shape[0] > 640 or image.shape[1] > 640 else image
        cv2.imshow("Label face-mask crop", preview)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key in LABELS:
            out_dir = dest / LABELS[key]
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / path.name
            if target.exists():
                target = out_dir / f"{path.stem}_{len(list(out_dir.iterdir()))}{path.suffix}"
            shutil.copy2(path, target)
    cv2.destroyAllWindows()
    print("Labelling session complete.")


if __name__ == "__main__":
    main()

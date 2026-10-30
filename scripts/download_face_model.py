from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

URLS = {
    "yolov8n-face": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
    "yolo26n-face": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Download a pretrained YOLO face detector checkpoint.")
    parser.add_argument("--model", choices=URLS, default="yolov8n-face")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{args.model}.pt"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"Model already exists: {target}")
        return
    print(f"Downloading {args.model} from the YOLO Face project...")
    urlretrieve(URLS[args.model], target)
    if target.stat().st_size < 1_000_000:
        target.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is unexpectedly small; download may have failed.")
    print(f"Saved: {target} ({target.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()

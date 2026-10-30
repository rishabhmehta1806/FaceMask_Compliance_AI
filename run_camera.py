from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from maskguard.config import load_config
from maskguard.pipeline import MaskCompliancePipeline


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Run real-time face-mask compliance monitoring.")
    parser.add_argument("--source", default="0", help="Webcam index (0) or path to a video file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--save", default="", help="Optional path for saving annotated video")
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = MaskCompliancePipeline(config)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        pipeline.close()
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, results = pipeline.process_frame(frame)
            if args.save and writer is None:
                h, w = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if writer is not None:
                writer.write(annotated)
            cv2.imshow("Explainable Face Mask Compliance Monitor | press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()

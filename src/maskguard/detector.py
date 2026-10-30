from __future__ import annotations

from pathlib import Path
import cv2


class FaceDetector:
    """YOLO face detector with an optional MediaPipe fallback."""

    def __init__(self, model_path: str | Path, confidence: float = 0.35,
                 image_size: int = 640, use_mediapipe_fallback: bool = True):
        self.confidence = confidence
        self.image_size = image_size
        self.yolo = None
        self.mp_detector = None
        self.backend = "unavailable"
        path = Path(model_path)
        if path.exists():
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(path))
                self.backend = "YOLO"
            except Exception as exc:
                print(f"[WARN] Could not load YOLO face weights ({exc}).")
        else:
            print(f"[WARN] YOLO face weights not found at {path}.")

        if self.yolo is None and use_mediapipe_fallback:
            try:
                import mediapipe as mp
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=confidence
                )
                self.backend = "MediaPipe fallback"
            except Exception as exc:
                print(f"[WARN] MediaPipe fallback unavailable ({exc}).")
        if self.backend == "unavailable":
            raise RuntimeError(
                "No face detector is available. Download YOLO face weights with "
                "python scripts/download_face_model.py, or install MediaPipe."
            )
        print(f"[INFO] Face detector backend: {self.backend}")

    def detect(self, frame_bgr) -> list[tuple[int, int, int, int]]:
        height, width = frame_bgr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        if self.yolo is not None:
            result = self.yolo.predict(frame_bgr, conf=self.confidence, imgsz=self.image_size, verbose=False)[0]
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box[:4]
                    boxes.append((
                        max(0, int(x1)), max(0, int(y1)),
                        min(width, int(x2)), min(height, int(y2)),
                    ))
            return boxes

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mp_detector.process(rgb)
        if result.detections:
            for detection in result.detections:
                relative = detection.location_data.relative_bounding_box
                x1 = int(relative.xmin * width)
                y1 = int(relative.ymin * height)
                x2 = int((relative.xmin + relative.width) * width)
                y2 = int((relative.ymin + relative.height) * height)
                boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
        return boxes

    def close(self) -> None:
        if self.mp_detector is not None:
            self.mp_detector.close()

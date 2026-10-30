from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class LandmarkResult:
    available: bool
    points: dict[str, tuple[int, int]]


class FaceLandmarker:
    """MediaPipe FaceMesh wrapper. Landmarks localize facial regions; they do not alone prove mask coverage."""

    # Stable FaceMesh indices for reference points.
    INDEX = {"nose_tip": 1, "upper_lip": 13, "lower_lip": 14, "left_mouth": 61, "right_mouth": 291, "chin": 152}

    def __init__(self, max_faces: int = 1):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt first.") from exc
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.45,
        )

    def detect(self, face_bgr: np.ndarray) -> LandmarkResult:
        if face_bgr is None or face_bgr.size == 0:
            return LandmarkResult(False, {})
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return LandmarkResult(False, {})
        face = result.multi_face_landmarks[0]
        height, width = face_bgr.shape[:2]
        points: dict[str, tuple[int, int]] = {}
        for name, idx in self.INDEX.items():
            if idx < len(face.landmark):
                point = face.landmark[idx]
                points[name] = (
                    int(np.clip(point.x * width, 0, width - 1)),
                    int(np.clip(point.y * height, 0, height - 1)),
                )
        return LandmarkResult(len(points) >= 4, points)

    def close(self) -> None:
        self._mesh.close()

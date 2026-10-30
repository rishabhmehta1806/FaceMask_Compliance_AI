from __future__ import annotations

from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class QualityResult:
    score: float
    blur_score: float
    illumination_score: float
    face_size_score: float
    landmark_score: float
    occlusion_risk: float
    blur_variance: float
    mean_brightness: float

    def to_dict(self) -> dict:
        return asdict(self)


def assess_image_quality(
    face_bgr: np.ndarray,
    landmarks_available: bool,
    blur_reference: float = 120.0,
    minimum_face_pixels: int = 72,
    face_box_size: tuple[int, int] | None = None,
) -> QualityResult:
    """Estimate image suitability; occlusion is a landmark-coverage proxy, not a mask detector."""
    if face_bgr is None or face_bgr.size == 0:
        return QualityResult(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip(blur_variance / max(blur_reference, 1.0), 0.0, 1.0))

    mean_brightness = float(gray.mean())
    # Full score close to mid-gray; progressively lower near under/over-exposure.
    illumination_score = float(np.clip(1.0 - abs(mean_brightness - 127.5) / 127.5, 0.0, 1.0))

    if face_box_size is None:
        height, width = gray.shape[:2]
    else:
        width, height = face_box_size
    short_side = min(int(width), int(height))
    face_size_score = float(np.clip((short_side - minimum_face_pixels * 0.5) / max(minimum_face_pixels, 1), 0.0, 1.0))

    landmark_score = 1.0 if landmarks_available else 0.35
    occlusion_risk = 0.15 if landmarks_available else 0.70

    score = (
        0.35 * blur_score
        + 0.25 * illumination_score
        + 0.25 * face_size_score
        + 0.15 * landmark_score
    )
    return QualityResult(
        score=float(np.clip(score, 0.0, 1.0)),
        blur_score=blur_score,
        illumination_score=illumination_score,
        face_size_score=face_size_score,
        landmark_score=landmark_score,
        occlusion_risk=occlusion_risk,
        blur_variance=blur_variance,
        mean_brightness=mean_brightness,
    )

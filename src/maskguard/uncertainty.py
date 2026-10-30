from __future__ import annotations
import numpy as np


def estimate_uncertainty(probabilities: np.ndarray, quality_score: float, uncertainty_threshold: float = 0.43, minimum_confidence: float = 0.58) -> tuple[float, bool, float]:
    """Entropy + low-confidence uncertainty, with a quality penalty."""
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    confidence = float(p.max())
    entropy = float(-(p * np.log(p)).sum() / np.log(max(len(p), 2)))
    uncertainty = float(np.clip(0.55 * entropy + 0.35 * (1.0 - confidence) + 0.10 * (1.0 - quality_score), 0, 1))
    return uncertainty, confidence < minimum_confidence or uncertainty > uncertainty_threshold or quality_score < 0.35, confidence

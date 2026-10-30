from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    history: deque = field(default_factory=deque)
    last_seen: float = 0.0
    improper_since: float | None = None
    alerted: bool = False


class TemporalTracker:
    def __init__(self, class_names: list[str], history_length: int = 12, ttl: float = 1.5,
                 iou_threshold: float = 0.25, alert_seconds: float = 5.0,
                 min_quality_for_alert: float = 0.45):
        self.class_names = class_names
        self.correct_idx = class_names.index("correctly_worn")
        self.history_length = history_length
        self.ttl = ttl
        self.iou_threshold = iou_threshold
        self.alert_seconds = alert_seconds
        self.min_quality_for_alert = min_quality_for_alert
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        # Greedy IoU association: appropriate for a lightweight student-project tracker.
        unmatched = set(self.tracks)
        for item in detections:
            best_id, best_iou = None, 0.0
            for track_id in unmatched:
                score = box_iou(item["bbox"], self.tracks[track_id].bbox)
                if score > best_iou:
                    best_id, best_iou = track_id, score
            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = Track(best_id, item["bbox"], deque(maxlen=self.history_length), now)
            else:
                unmatched.remove(best_id)

            track = self.tracks[best_id]
            track.bbox = item["bbox"]
            track.last_seen = now
            track.history.append({"probabilities": item["probabilities"], "quality": item["quality_score"],
                                  "confidence": item["confidence"], "uncertainty": item["uncertainty"],
                                  "uncertain": item["uncertain"]})
            matrix = np.stack([x["probabilities"] for x in track.history])
            average = matrix.mean(axis=0)
            pred_idx = int(np.argmax(average))
            labels = np.argmax(matrix, axis=1)
            temporal_consistency = float(np.mean(labels == pred_idx))
            correct_ratio = float(np.mean(labels == self.correct_idx))
            avg_quality = float(np.mean([x["quality"] for x in track.history]))
            avg_confidence = float(np.mean([x["confidence"] for x in track.history]))
            avg_uncertainty = float(np.mean([x["uncertainty"] for x in track.history]))
            has_uncertain_frame = bool(item["uncertain"])

            # Compliance remains low for a stable, consistently incorrect prediction.
            p_correct = float(average[self.correct_idx])
            score = 100.0 * (0.50 * p_correct + 0.20 * correct_ratio + 0.20 * avg_quality + 0.10 * avg_confidence)

            is_improper = pred_idx != self.correct_idx
            alert = False
            if is_improper and not has_uncertain_frame and avg_quality >= self.min_quality_for_alert:
                if track.improper_since is None:
                    track.improper_since = now
                if now - track.improper_since >= self.alert_seconds and not track.alerted:
                    alert = True
                    track.alerted = True
            elif pred_idx == self.correct_idx:
                track.improper_since = None
                track.alerted = False
            else:
                # Low quality or uncertain evidence pauses and resets the alert timer.
                track.improper_since = None

            item.update({
                "track_id": best_id,
                "probabilities_temporal": average,
                "predicted_index": pred_idx,
                "predicted_class": self.class_names[pred_idx],
                "temporal_consistency": temporal_consistency,
                "temporal_correct_ratio": correct_ratio,
                "compliance_score": float(np.clip(score, 0, 100)),
                "mean_uncertainty": avg_uncertainty,
                "mean_confidence": avg_confidence,
                "decision": "REVIEW" if has_uncertain_frame else self.class_names[pred_idx].replace("_", " ").upper(),
                "alert": alert,
                "improper_duration": max(0.0, now - track.improper_since) if track.improper_since is not None else 0.0,
            })

        for track_id in list(self.tracks):
            if now - self.tracks[track_id].last_seen > self.ttl:
                del self.tracks[track_id]
        return detections

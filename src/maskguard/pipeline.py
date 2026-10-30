from __future__ import annotations

from pathlib import Path
import csv
import time
import cv2

from .classifier import MaskClassifier
from .config import resolve_path
from .detector import FaceDetector
from .explainability import make_explanation, draw_landmarks
from .landmarks import FaceLandmarker
from .quality import assess_image_quality
from .temporal import TemporalTracker
from .uncertainty import estimate_uncertainty


class MaskCompliancePipeline:
    def __init__(self, config: dict):
        detector_cfg = config.get("detector", {})
        classifier_cfg = config.get("classifier", {})
        quality_cfg = config.get("quality", {})
        tracking_cfg = config.get("tracking", {})

        self.detector = FaceDetector(
            resolve_path(detector_cfg.get("model_path", "models/yolov8n-face.pt")),
            confidence=float(detector_cfg.get("confidence", 0.35)),
            image_size=int(detector_cfg.get("image_size", 640)),
            use_mediapipe_fallback=bool(detector_cfg.get("use_mediapipe_fallback", True)),
        )
        self.classifier = MaskClassifier(
            resolve_path(classifier_cfg.get("checkpoint_path", "models/best_mask_classifier.pth")),
            image_size=int(classifier_cfg.get("image_size", 224)),
        )
        self.landmarker = FaceLandmarker(max_faces=1)
        self.tracker = TemporalTracker(
            class_names=self.classifier.class_names,
            history_length=int(tracking_cfg.get("history_length", 12)),
            ttl=float(tracking_cfg.get("track_ttl_seconds", 1.5)),
            iou_threshold=float(tracking_cfg.get("iou_match_threshold", 0.25)),
            alert_seconds=float(tracking_cfg.get("alert_after_seconds", 5.0)),
            min_quality_for_alert=float(tracking_cfg.get("minimum_quality_for_alert", 0.45)),
        )
        self.quality_cfg = quality_cfg
        self.classifier_cfg = classifier_cfg
        alerts_cfg = config.get("alerts", {})
        self.alert_csv = resolve_path(alerts_cfg.get("csv_path", "logs/alerts.csv"))
        self.alert_csv.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_alert_csv()

    def _ensure_alert_csv(self) -> None:
        if not self.alert_csv.exists():
            with self.alert_csv.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["timestamp", "track_id", "predicted_class", "compliance_score", "confidence", "reason"])

    def _log_alert(self, item: dict, reason: str) -> None:
        with self.alert_csv.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), item["track_id"], item["predicted_class"],
                f"{item['compliance_score']:.1f}", f"{item['confidence']:.3f}", reason,
            ])

    @staticmethod
    def _expanded_crop(frame, bbox, margin: float = 0.16):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        pad_x, pad_y = int(width * margin), int(height * margin)
        h, w = frame.shape[:2]
        ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        ex2, ey2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return frame[ey1:ey2, ex1:ex2], (ex1, ey1)

    def process_frame(self, frame_bgr, timestamp: float | None = None):
        now = timestamp if timestamp is not None else time.monotonic()
        detections = []
        for bbox in self.detector.detect(frame_bgr):
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            crop, origin = self._expanded_crop(frame_bgr, bbox)
            if crop.size == 0:
                continue
            landmark_result = self.landmarker.detect(crop)
            q = assess_image_quality(
                crop,
                landmarks_available=landmark_result.available,
                blur_reference=float(self.quality_cfg.get("blur_reference", 120.0)),
                minimum_face_pixels=int(self.quality_cfg.get("minimum_face_pixels", 72)),
                face_box_size=(x2 - x1, y2 - y1),
            )
            probabilities = self.classifier.predict(crop)
            uncertainty, uncertain, confidence = estimate_uncertainty(
                probabilities, q.score,
                uncertainty_threshold=float(self.classifier_cfg.get("uncertainty_threshold", 0.43)),
                minimum_confidence=float(self.classifier_cfg.get("minimum_confidence", 0.58)),
            )
            detections.append({
                "bbox": bbox,
                "crop": crop,
                "crop_origin": origin,
                "probabilities": probabilities,
                "quality_score": q.score,
                "quality": q.to_dict(),
                "landmarks": landmark_result.points,
                "landmarks_available": landmark_result.available,
                "uncertainty": uncertainty,
                "uncertain": uncertain or q.score < float(self.quality_cfg.get("minimum_score_for_decision", 0.35)),
                "confidence": confidence,
            })

        tracked = self.tracker.update(detections, now)
        output = frame_bgr.copy()
        for item in tracked:
            x1, y1, x2, y2 = item["bbox"]
            label = item["decision"]
            color = (0, 165, 255) if label == "REVIEW" else ((0, 180, 0) if item["predicted_class"] == "correctly_worn" else (0, 0, 255))
            if item["alert"]:
                color = (0, 0, 255)
            reason = make_explanation(item["predicted_class"], item["landmarks"], item["quality"], item["mean_uncertainty"])
            item["reason"] = reason
            if item["alert"]:
                self._log_alert(item, reason)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"ID {item['track_id']} | {label} | Score {item['compliance_score']:.0f}"
            cv2.putText(output, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            if item["alert"]:
                cv2.putText(output, "MASK COMPLIANCE ALERT", (x1, min(output.shape[0] - 8, y2 + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            reason_line = reason[:64]
            cv2.putText(output, reason_line, (x1, min(output.shape[0] - 8, y2 + 42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            # Draw key landmarks in the original frame coordinates.
            ox, oy = item["crop_origin"]
            for name, (px, py) in item["landmarks"].items():
                cv2.circle(output, (ox + px, oy + py), 3, (0, 255, 255), -1)
        return output, tracked

    def close(self) -> None:
        self.detector.close()
        self.landmarker.close()

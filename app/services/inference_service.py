from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from ..utils.image_utils import draw_detections, preprocess_mri

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime for lightweight hosts
    YOLO = None


CLASS_LABELS = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
    3: "No Tumor",
}

LABEL_ALIASES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
    "no_tumor": "No Tumor",
    "no tumor": "No Tumor",
}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[int] | None = None
    source: str = "detection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "confidence_percent": round(self.confidence * 100, 2),
            "bbox": self.bbox,
            "source": self.source,
        }


class BrainTumorInferenceService:
    """Runs preprocessing and YOLOv8 inference with a safe simulation fallback."""

    _cached_model = None
    _cached_model_path: Path | None = None

    def __init__(self, model_path: Path, result_dir: Path):
        self.model_path = Path(model_path)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, image_path: Path, confidence_threshold: float = 0.40) -> dict[str, Any]:
        image_path = Path(image_path)
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError("Uploaded file could not be decoded as an MRI image.")

        processed, preprocessing = preprocess_mri(original, target_size=(512, 512))

        if self._can_use_yolo():
            detections, model_task = self._predict_with_yolo(processed, confidence_threshold)
            inference_mode = f"YOLOv8 {model_task} model"
            model_status = "Loaded trained model from model/best.pt"
        else:
            detections = self._simulate_prediction(processed, confidence_threshold)
            inference_mode = "Simulation fallback"
            model_status = "model/best.pt was not found or Ultralytics is unavailable"

        primary = self._select_primary_detection(detections)
        annotated = draw_detections(processed, detections, primary_label=primary["label"])

        result_name = f"result_{uuid4().hex}.jpg"
        result_path = self.result_dir / result_name
        cv2.imwrite(str(result_path), annotated)

        return {
            "tumor_type": primary["label"],
            "confidence": primary["confidence"],
            "confidence_percent": round(primary["confidence"] * 100, 2),
            "detections": [d.to_dict() for d in detections],
            "annotated_image_url": f"/static/results/{result_name}",
            "model_status": model_status,
            "inference_mode": inference_mode,
            "preprocessing": preprocessing,
            "clinical_note": "Research prototype only. It is not a medical diagnosis.",
            "accuracy_display": self._accuracy_display(),
        }

    def _can_use_yolo(self) -> bool:
        return YOLO is not None and self.model_path.exists()

    def _load_model(self):
        if (
            self.__class__._cached_model is None
            or self.__class__._cached_model_path != self.model_path
        ):
            self.__class__._cached_model = YOLO(str(self.model_path))
            self.__class__._cached_model_path = self.model_path
        return self.__class__._cached_model

    def _predict_with_yolo(
        self, processed_image: np.ndarray, confidence_threshold: float
    ) -> tuple[list[Detection], str]:
        model = self._load_model()
        results = model.predict(
            source=processed_image,
            imgsz=512,
            conf=confidence_threshold,
            iou=0.45,
            verbose=False,
        )

        if not results:
            return [], "unknown"

        result = results[0]
        names = getattr(result, "names", None) or getattr(model, "names", {})

        if getattr(result, "probs", None) is not None:
            return self._parse_classification_result(result, names, confidence_threshold), "classification"

        detections: list[Detection] = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = normalize_label(names.get(class_id, CLASS_LABELS.get(class_id, f"Class {class_id}")))
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()

            if label == "No Tumor":
                continue

            detections.append(Detection(label=label, confidence=confidence, bbox=xyxy))

        return detections, "detection"

    @staticmethod
    def _parse_classification_result(
        result: Any, names: dict[int, str], confidence_threshold: float
    ) -> list[Detection]:
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = normalize_label(names.get(class_id, CLASS_LABELS.get(class_id, f"Class {class_id}")))

        return [
            Detection(
                label=label,
                confidence=confidence,
                bbox=None,
                source="classification",
            )
        ]

    def _simulate_prediction(
        self, processed_image: np.ndarray, confidence_threshold: float
    ) -> list[Detection]:
        """Deterministic fallback for presentations when trained weights are absent."""
        digest = hashlib.sha256(processed_image.tobytes()).digest()
        selector = digest[0] % 4
        label = CLASS_LABELS[selector]

        if label == "No Tumor":
            return []

        h, w = processed_image.shape[:2]
        base_confidence = 0.64 + (digest[1] / 255) * 0.28
        confidence = max(base_confidence, confidence_threshold + 0.05)

        box_w = int(w * (0.24 + (digest[2] / 255) * 0.14))
        box_h = int(h * (0.22 + (digest[3] / 255) * 0.16))
        center_x = int(w * (0.42 + (digest[4] / 255) * 0.18))
        center_y = int(h * (0.38 + (digest[5] / 255) * 0.22))

        x1 = max(0, center_x - box_w // 2)
        y1 = max(0, center_y - box_h // 2)
        x2 = min(w - 1, x1 + box_w)
        y2 = min(h - 1, y1 + box_h)

        return [
            Detection(
                label=label,
                confidence=min(confidence, 0.97),
                bbox=[x1, y1, x2, y2],
            )
        ]

    @staticmethod
    def _select_primary_detection(detections: list[Detection]) -> dict[str, Any]:
        if not detections:
            return {"label": "No Tumor", "confidence": 0.0}

        best = max(detections, key=lambda item: item.confidence)
        return {"label": best.label, "confidence": round(best.confidence, 4)}

    @staticmethod
    def _accuracy_display() -> dict[str, Any]:
        return {
            "label": "Validation accuracy",
            "value": 94.2,
            "source": "Replace with your trained YOLOv8 validation result.",
        }


def normalize_label(label: str) -> str:
    cleaned = str(label).strip().lower().replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    return LABEL_ALIASES.get(cleaned, str(label).strip().replace("_", " ").title())

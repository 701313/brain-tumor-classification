from __future__ import annotations

import cv2
import numpy as np


COLORS = {
    "Glioma": (43, 122, 255),
    "Meningioma": (41, 191, 143),
    "Pituitary": (187, 92, 255),
    "No Tumor": (120, 130, 140),
}


def preprocess_mri(image: np.ndarray, target_size: tuple[int, int] = (512, 512)):
    """Resize, denoise, normalize, and lightly enhance MRI contrast."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    denoised = cv2.GaussianBlur(resized, (5, 5), 0)

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    preprocessing = {
        "resize": f"{target_size[0]}x{target_size[1]}",
        "denoising": "Gaussian blur 5x5",
        "normalization": "OpenCV min-max normalization to 0-255",
        "roi_extraction": "Full brain slice used; add skull stripping for clinical ROI.",
    }
    return normalized, preprocessing


def draw_detections(image: np.ndarray, detections: list, primary_label: str) -> np.ndarray:
    annotated = image.copy()

    if not detections:
        cv2.putText(
            annotated,
            "No Tumor Detected",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLORS["No Tumor"],
            2,
            cv2.LINE_AA,
        )
        return annotated

    for detection in detections:
        label = detection.label
        confidence = detection.confidence
        color = COLORS.get(label, (50, 210, 255))
        if not detection.bbox:
            text = f"{label} {confidence * 100:.1f}%"
            cv2.rectangle(annotated, (16, 18), (16 + 330, 70), color, -1)
            cv2.putText(
                annotated,
                text,
                (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            continue

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {confidence * 100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        y_text = max(26, y1 - 8)
        cv2.rectangle(
            annotated,
            (x1, y_text - text_h - 10),
            (x1 + text_w + 12, y_text + 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 6, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Primary: {primary_label}",
        (24, annotated.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def normalize_label(label: str) -> str:
    aliases = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "notumor": "No Tumor",
        "no_tumor": "No Tumor",
        "no tumor": "No Tumor",
    }
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    return aliases.get(key, label.replace("_", " ").title())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one MRI image with model/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="model/best.pt")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(model_path))
    result = model.predict(str(image_path), imgsz=args.imgsz, verbose=False)[0]

    if result.probs is not None:
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = normalize_label(result.names[class_id])
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")
        return

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("Prediction: No Tumor")
        print("Confidence: 0.00%")
        return

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])
    label = normalize_label(result.names[class_id])
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Bounding box: {best_box.xyxy[0].detach().cpu().numpy().astype(int).tolist()}")


if __name__ == "__main__":
    main()

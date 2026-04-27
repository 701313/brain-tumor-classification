from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO


CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no_tumor",
    "pituitary": "pituitary",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 classification model from the MRI archive.zip dataset."
    )
    parser.add_argument("--zip", required=True, help="Path to archive.zip")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--project", default="runs/brain_tumor_cls")
    parser.add_argument("--name", default="yolov8_mri_classifier")
    parser.add_argument("--export-to", default="model/best.pt")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "datasets" / "archive_raw"
    prepared_dir = project_root / "datasets" / "brain_tumor_cls"

    extract_archive(zip_path, raw_dir)
    prepare_classification_dataset(raw_dir, prepared_dir, args.val_ratio, args.seed)
    print_dataset_summary(prepared_dir)

    if args.prepare_only:
        print(f"Dataset prepared at: {prepared_dir}")
        return

    model = YOLO(args.model)
    model.train(
        data=str(prepared_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        degrees=7,
        translate=0.06,
        scale=0.18,
        fliplr=0.5,
        project=args.project,
        name=args.name,
    )

    trained_model_path = project_root / args.project / args.name / "weights" / "best.pt"
    export_path = project_root / args.export_to
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_model_path, export_path)
    print(f"Copied trained model to: {export_path}")

    trained_model = YOLO(str(export_path))
    trained_model.val(data=str(prepared_dir), split="test", imgsz=args.imgsz)


def extract_archive(zip_path: Path, raw_dir: Path) -> None:
    marker = raw_dir / ".extracted"
    if marker.exists():
        print(f"Using existing extraction: {raw_dir}")
        return

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {raw_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    marker.write_text(str(zip_path), encoding="utf-8")


def prepare_classification_dataset(
    raw_dir: Path, prepared_dir: Path, val_ratio: float, seed: int
) -> None:
    marker = prepared_dir / ".prepared"
    if marker.exists():
        print(f"Using existing prepared dataset: {prepared_dir}")
        return

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    rng = random.Random(seed)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for source_class, target_class in CLASS_MAP.items():
        train_source = raw_dir / "Training" / source_class
        test_source = raw_dir / "Testing" / source_class
        if not train_source.exists() or not test_source.exists():
            raise FileNotFoundError(
                f"Expected Training/Testing folders for class '{source_class}' inside {raw_dir}"
            )

        train_images = sorted(
            image for image in train_source.iterdir() if image.suffix.lower() in image_extensions
        )
        rng.shuffle(train_images)
        val_count = max(1, int(len(train_images) * val_ratio))
        val_images = train_images[:val_count]
        final_train_images = train_images[val_count:]

        copy_images(final_train_images, prepared_dir / "train" / target_class)
        copy_images(val_images, prepared_dir / "val" / target_class)
        copy_images(
            sorted(image for image in test_source.iterdir() if image.suffix.lower() in image_extensions),
            prepared_dir / "test" / target_class,
        )

    marker.write_text("prepared", encoding="utf-8")


def copy_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, destination / image.name)


def print_dataset_summary(dataset_dir: Path) -> None:
    print("\nDataset summary")
    for split in ("train", "val", "test"):
        for class_dir in sorted((dataset_dir / split).iterdir()):
            if class_dir.is_dir():
                count = sum(1 for item in class_dir.iterdir() if item.is_file())
                print(f"{split:5s} {class_dir.name:12s} {count}")


if __name__ == "__main__":
    main()

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="training/data.yaml",
        epochs=80,
        imgsz=512,
        batch=16,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        degrees=8,
        translate=0.08,
        scale=0.25,
        shear=2,
        fliplr=0.5,
        mosaic=0.4,
        mixup=0.05,
        project="runs/brain_tumor",
        name="yolov8_brain_tumor",
    )
    model.val(data="training/data.yaml", imgsz=512, conf=0.25, iou=0.45)


if __name__ == "__main__":
    main()

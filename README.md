# Brain Tumor Detection AI Web Application

Production-style Flask + YOLOv8 implementation for an IEEE-inspired final year project:
**A Multi-stage Hybrid Deep Learning Model for Brain Tumor Classification**.

The app accepts MRI brain scans, preprocesses the image, localizes tumors with bounding boxes,
classifies the tumor type, and returns an annotated result image plus a downloadable report.

## 1. System Architecture Explanation

```text
Browser UI
  -> Flask route /api/predict
  -> upload validation
  -> OpenCV preprocessing
  -> YOLOv8 inference with confidence threshold and built-in NMS
  -> result annotation
  -> report generation
  -> JSON response to frontend
```

Core modules:

- `app/routes/main.py`: HTTP pages, prediction API, report download, health check.
- `app/services/inference_service.py`: YOLOv8 loading, inference parsing, deterministic fallback.
- `app/utils/image_utils.py`: resize, denoise, normalize, contrast enhancement, annotation.
- `app/utils/file_utils.py`: upload validation and secure file naming.
- `static/script.js`: upload flow, threshold slider, dark mode, multilingual UI.

Medical note: this is an academic research prototype. It must not be used as a standalone
diagnostic device.

## 2. Folder Structure

```text
brain-tumor-app/
├── app/
│   ├── routes/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference_service.py
│   │   └── report_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   └── image_utils.py
│   └── __init__.py
├── model/
│   ├── README.md
│   └── best.pt
├── reports/
├── static/
│   ├── results/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── training/
│   ├── data.yaml
│   └── train_yolo.py
├── uploads/
├── app.py
├── run.py
├── requirements.txt
└── README.md
```

## 3. Backend Code

Backend code is implemented in:

- `app.py`: WSGI entrypoint for hosting platforms.
- `run.py`: local development runner.
- `app/__init__.py`: Flask application factory.
- `app/routes/main.py`: routes and API handlers.

Important endpoints:

- `GET /`: web UI.
- `POST /api/predict`: accepts multipart upload field `image` and `threshold`.
- `GET /api/report/<filename>`: downloads generated text report.
- `GET /health`: returns service and model status.

## 4. Frontend Code

Frontend code is implemented in:

- `templates/index.html`
- `static/style.css`
- `static/script.js`

Included features:

- Responsive medical dashboard layout.
- Dark mode.
- Confidence threshold slider.
- Preview before upload.
- Annotated image display.
- Result image download.
- Report download.
- English, Hindi, and Spanish UI text.
- Accuracy display placeholder that should be replaced with your trained model metric.

## 5. Model Integration Code

Place trained YOLOv8 weights at:

```text
model/best.pt
```

The app loads this file with:

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
```

Inference settings:

- `imgsz=512`
- confidence threshold from the UI slider
- `iou=0.45`
- Ultralytics YOLO applies Non-Max Suppression internally

If the model is unavailable, the app uses deterministic simulated predictions so the full
web workflow still runs during development, demos, and deployment testing.

## 6. requirements.txt

```text
Flask==3.0.3
Werkzeug==3.0.3
opencv-python-headless==4.10.0.84
numpy==1.26.4
ultralytics==8.3.40
Pillow==10.4.0
gunicorn==22.0.0
```

## 7. Local Setup Guide

```bash
cd brain-tumor-app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

To use your trained model:

```text
copy your YOLOv8 weights to model/best.pt
restart Flask
```

## 8. Deployment Guide

### Hostinger Shared Hosting / hPanel

As of April 24, 2026, Hostinger's official help center says Flask and Python projects are
supported on Hostinger VPS, not on regular shared/web hPanel hosting, because Python package
management and runtime setup require root/server access.

Official references:

- Hostinger Flask support: https://support.hostinger.com/en/articles/9791148-is-flask-supported-at-hostinger
- Hostinger Python support: https://support.hostinger.com/en/articles/3648030-is-python-supported-at-hostinger
- Hostinger VPS details: https://support.hostinger.com/en/articles/1583582-how-and-why-to-purchase-vps-hosting

What you can do on shared hosting:

1. Log in to hPanel.
2. Open `Websites -> Manage -> File Manager`.
3. Upload only static frontend files if you want a non-AI landing page.
4. Host the Flask AI backend somewhere else, such as Render, Railway, AWS, or Hostinger VPS.
5. Update `fetch("/api/predict")` in `static/script.js` to the hosted backend URL.

This is not recommended for the final project because browser-only hosting cannot run YOLOv8.

### Hostinger VPS Recommended Path

1. Buy or open a Hostinger VPS plan from hPanel.
2. Choose Ubuntu 22.04 or 24.04.
3. Connect by SSH:

```bash
ssh root@YOUR_SERVER_IP
```

4. Install system packages:

```bash
apt update
apt install -y python3 python3-venv python3-pip nginx git
```

5. Upload the project by Git, SCP, or hPanel VPS file tools:

```bash
git clone YOUR_REPOSITORY_URL /opt/brain-tumor-app
cd /opt/brain-tumor-app
```

6. Create virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

7. Add your trained model:

```bash
mkdir -p model
scp best.pt root@YOUR_SERVER_IP:/opt/brain-tumor-app/model/best.pt
```

8. Test with Gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

9. Create a systemd service:

```ini
[Unit]
Description=Brain Tumor Flask App
After=network.target

[Service]
User=root
WorkingDirectory=/opt/brain-tumor-app
Environment="PATH=/opt/brain-tumor-app/.venv/bin"
ExecStart=/opt/brain-tumor-app/.venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Save as:

```text
/etc/systemd/system/brain-tumor.service
```

10. Start service:

```bash
systemctl daemon-reload
systemctl enable brain-tumor
systemctl start brain-tumor
systemctl status brain-tumor
```

11. Configure Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    client_max_body_size 8M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

12. Enable site and SSL:

```bash
nginx -t
systemctl restart nginx
apt install -y certbot python3-certbot-nginx
certbot --nginx -d your-domain.com -d www.your-domain.com
```

### Render

1. Push the repo to GitHub.
2. Create a new Render Web Service.
3. Set build command:

```bash
pip install -r requirements.txt
```

4. Set start command:

```bash
gunicorn app:app
```

5. Use a paid instance for large YOLO weights and persistent performance.

### Railway

1. Push the repo to GitHub.
2. Create a Railway project from the repository.
3. Add start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

4. Upload `best.pt` through storage, repository LFS, or deployment artifact.

### AWS

Recommended options:

- EC2 Ubuntu server with Nginx + Gunicorn for simple deployment.
- ECS/Fargate with Docker for production scaling.
- S3 for storing uploaded and annotated images.
- CloudWatch for logs.
- Application Load Balancer and HTTPS certificate for public release.

For GPU inference, use an EC2 GPU instance or move inference to a dedicated model endpoint.

## 9. AI Model Training

### Use Your `archive.zip` Dataset

Your archive has this classification structure:

```text
archive.zip
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

This dataset does not include bounding-box annotation `.txt` files, so it can train a
YOLOv8 classification model, not a true YOLOv8 detection/localization model. The web app
supports that classifier and will show the predicted class and confidence. For real bounding
boxes, annotate the MRI images in YOLO detection format using a tool such as Roboflow, CVAT,
or LabelImg, then train with `training/train_yolo.py`.

Prepare the dataset only:

```powershell
cd C:\major\brain-tumor-app
.\.venv\Scripts\python.exe training\train_classifier_from_zip.py --zip "C:\Users\pinni\OneDrive\Desktop\archive.zip" --prepare-only
```

Train, test, and copy the trained model into `model/best.pt`:

```powershell
cd C:\major\brain-tumor-app
.\.venv\Scripts\python.exe training\train_classifier_from_zip.py --zip "C:\Users\pinni\OneDrive\Desktop\archive.zip" --epochs 40 --batch 16 --imgsz 224
```

For a quick CPU smoke test, run only one epoch:

```powershell
cd C:\major\brain-tumor-app
.\.venv\Scripts\python.exe training\train_classifier_from_zip.py --zip "C:\Users\pinni\OneDrive\Desktop\archive.zip" --epochs 1 --batch 8 --imgsz 224 --name smoke_test
```

Predict one test image from the terminal after training:

```powershell
cd C:\major\brain-tumor-app
.\.venv\Scripts\python.exe training\predict_image.py --image "datasets\brain_tumor_cls\test\glioma\Te-gl_0010.jpg"
```

Run the web app with the trained classifier:

```powershell
cd C:\major\brain-tumor-app
.\.venv\Scripts\python.exe run.py
```

Open:

```text
http://127.0.0.1:5000
```

### Dataset Structure

Use YOLO detection format:

```text
datasets/brain_tumor_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label file must match the image name:

```text
images/train/mri_001.jpg
labels/train/mri_001.txt
```

YOLO label format:

```text
class_id x_center y_center width height
```

Values are normalized between `0` and `1`.

Classes:

- `0`: Glioma
- `1`: Meningioma
- `2`: Pituitary
- `3`: No Tumor

For pure no-tumor images, either use empty label files or train a classification branch
separately. For YOLO detection, empty labels are commonly used to teach background/no-object.

### Training Steps

```bash
cd brain-tumor-app
pip install -r requirements.txt
python training/train_yolo.py
```

Recommended starting configuration:

- Model: `yolov8n.pt` for laptops, `yolov8s.pt` or `yolov8m.pt` for stronger GPUs.
- Image size: `512`.
- Epochs: `80` to `120`.
- Batch size: `16`, reduce to `8` or `4` if GPU memory is limited.
- Optimizer: AdamW.
- Augmentation: rotation, translation, scale, horizontal flip, light mosaic, light mixup.

After training, copy:

```text
runs/brain_tumor/yolov8_brain_tumor/weights/best.pt
```

to:

```text
model/best.pt
```

### Evaluation Metrics

Track these in the Ultralytics validation output:

- Precision: among predicted tumors, how many are correct.
- Recall: among actual tumors, how many were detected.
- mAP50: mean average precision at IoU 0.50.
- mAP50-95: stricter COCO-style average over multiple IoU thresholds.
- Confusion matrix: class-level mistakes between glioma, meningioma, and pituitary.

For final-year reporting, include:

- train/validation loss curves.
- PR curve.
- confusion matrix.
- example predictions with bounding boxes.
- comparison table with the IEEE paper baseline or your reproduced baseline.

## 10. Future Improvements

- Replace fallback simulation with trained `best.pt`.
- Add skull stripping or brain ROI extraction.
- Add DICOM support with `pydicom`.
- Add user authentication and doctor/patient roles.
- Store history in PostgreSQL.
- Generate PDF reports.
- Add Grad-CAM or heatmap explanations.
- Add Dockerfile and CI tests.
- Add external validation using another MRI dataset.
- Calibrate confidence scores before clinical research use.

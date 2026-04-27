from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def create_report(result: dict, report_dir: Path) -> Path:
    """Create a lightweight downloadable clinical-style text report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    path = report_dir / f"brain_tumor_report_{uuid4().hex}.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detections = result.get("detections") or []
    detection_lines = []
    for index, detection in enumerate(detections, start=1):
        bbox = detection.get("bbox") or "not available for classification model"
        detection_lines.append(
            f"{index}. {detection['label']} | "
            f"confidence={detection['confidence_percent']}% | "
            f"bbox={bbox}"
        )
    if not detection_lines:
        detection_lines.append("No tumor bounding box was detected above the threshold.")

    content = f"""Brain Tumor Detection Report
Generated: {timestamp}

Predicted class: {result.get('tumor_type')}
Confidence: {result.get('confidence_percent')}%
Inference mode: {result.get('inference_mode')}
Model status: {result.get('model_status')}

Detections:
{chr(10).join(detection_lines)}

Preprocessing:
- Resize: {result.get('preprocessing', {}).get('resize')}
- Denoising: {result.get('preprocessing', {}).get('denoising')}
- Normalization: {result.get('preprocessing', {}).get('normalization')}

Important:
This system is a research prototype for academic use and must not be used as a
standalone medical diagnosis. A qualified radiologist must review all findings.
"""
    path.write_text(content, encoding="utf-8")
    return path

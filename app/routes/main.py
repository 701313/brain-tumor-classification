from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from ..services.inference_service import BrainTumorInferenceService
from ..services.report_service import create_report
from ..utils.file_utils import save_upload

main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.post("/api/predict")
def predict():
    threshold = _parse_threshold(request.form.get("threshold", "0.40"))

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No MRI image was uploaded."}), 400

    upload = request.files["image"]
    try:
        saved_path = save_upload(
            upload,
            current_app.config["UPLOAD_FOLDER"],
            current_app.config["ALLOWED_EXTENSIONS"],
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    service = BrainTumorInferenceService(
        model_path=current_app.config["MODEL_PATH"],
        result_dir=current_app.config["RESULT_FOLDER"],
    )
    try:
        result = service.predict(saved_path, confidence_threshold=threshold)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    report_path = create_report(result, current_app.config["REPORT_FOLDER"])
    result["report_url"] = f"/api/report/{Path(report_path).name}"

    return jsonify({"ok": True, "result": result})


@main_bp.get("/api/report/<path:filename>")
def report(filename):
    report_path = Path(current_app.config["REPORT_FOLDER"]) / Path(filename).name
    if not report_path.exists():
        return jsonify({"ok": False, "error": "Report not found."}), 404

    return send_file(report_path, as_attachment=True)


@main_bp.get("/health")
def health():
    model_path = Path(current_app.config["MODEL_PATH"])
    return jsonify(
        {
            "ok": True,
            "model_available": model_path.exists(),
            "model_path": str(model_path),
        }
    )


@main_bp.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify({"ok": False, "error": "File is too large. Maximum size is 8 MB."}), 413


def _parse_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 0.40
    return min(max(threshold, 0.05), 0.95)

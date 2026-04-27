from pathlib import Path

from flask import Flask

from .routes.main import main_bp


def create_app() -> Flask:
    """Application factory used by local development and WSGI hosts."""
    project_root = Path(__file__).resolve().parent.parent

    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )

    app.config.update(
        SECRET_KEY="change-this-secret-key-in-production",
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        PROJECT_ROOT=project_root,
        UPLOAD_FOLDER=project_root / "uploads",
        RESULT_FOLDER=project_root / "static" / "results",
        REPORT_FOLDER=project_root / "reports",
        MODEL_PATH=project_root / "model" / "best.pt",
        ALLOWED_EXTENSIONS={"png", "jpg", "jpeg", "bmp", "webp"},
    )

    for folder_key in ("UPLOAD_FOLDER", "RESULT_FOLDER", "REPORT_FOLDER"):
        Path(app.config[folder_key]).mkdir(parents=True, exist_ok=True)

    app.register_blueprint(main_bp)
    return app

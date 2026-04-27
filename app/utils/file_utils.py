from pathlib import Path
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def save_upload(file: FileStorage, upload_dir: Path, allowed_extensions: set[str]) -> Path:
    if not file or not file.filename:
        raise ValueError("Please select an MRI image file.")

    if not allowed_file(file.filename, allowed_extensions):
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Invalid file type. Allowed formats: {allowed}.")

    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    path = upload_dir / f"{uuid4().hex}.{extension}"
    file.save(path)
    return path

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_create_app():
    """Load the package safely even when shared hosting imports this file as app.py."""
    root = Path(__file__).resolve().parent
    package_dir = root / "app"
    spec = importlib.util.spec_from_file_location(
        "brain_tumor_backend",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Flask application package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["brain_tumor_backend"] = module
    spec.loader.exec_module(module)
    return module.create_app


create_app = _load_create_app()
app = create_app()
application = app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

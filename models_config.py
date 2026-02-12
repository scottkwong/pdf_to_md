"""
Load models.json from package data or filesystem.

Resolves models.json whether running from a clone (dev) or from an
installed package (pip/pipx).
"""
import json
import os
from typing import Dict


def _find_models_path() -> str:
    """Return path to models.json, for dev or installed package.

    Returns:
        Absolute path to models.json.

    Raises:
        FileNotFoundError: If models.json cannot be found.
    """
    import sys

    # Installed: models.json is in share/pdf-to-md/ (data-files)
    installed = os.path.join(sys.prefix, "share", "pdf-to-md", "models.json")
    if os.path.exists(installed):
        return installed

    # Dev: models.json is in repo root (next to this module)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "models.json")
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError("models.json not found")


def load_models_config() -> Dict:
    """Load and parse models.json configuration file.

    Returns:
        Dictionary of model configurations.

    Raises:
        FileNotFoundError: If models.json is not found.
        json.JSONDecodeError: If models.json is invalid JSON.
    """
    models_path = _find_models_path()
    with open(models_path, "r") as f:
        return json.load(f)

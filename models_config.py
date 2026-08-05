"""
Load models.json from package data or filesystem.

Resolves models.json whether running from a clone (dev) or from an
installed package (pip/pipx). Also defines the Provider and ModelKey
enums so code and tests reference models by autocompletable identifier
instead of retyping the exact strings.
"""
import json
import os
from enum import Enum
from typing import Dict


class Provider(str, Enum):
    """Canonical provider names as used in models.json and API-key lookup.

    A ``str`` subclass, so members compare equal to (and format as) their
    plain string values: ``Provider.FIREWORKS == "fireworks"``.
    """

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    FIREWORKS = "fireworks"

    __str__ = str.__str__  # f"{Provider.OPENAI}" -> "openai", not the repr


class ModelKey(str, Enum):
    """models.json keys, one member per entry.

    Kept in lockstep with models.json by test_models_config; add a member
    here whenever a model is added there (and vice versa).
    """

    GPT_5_5 = "gpt-5.5"
    GPT_5_5_PRO = "gpt-5.5-pro"
    GPT_5_4 = "gpt-5.4"
    GPT_5_2 = "gpt-5.2"
    OPENAI_GPT4O = "openai-gpt4o"
    GEMINI_3_FLASH = "gemini-3-flash"
    GEMINI_3_PRO = "gemini-3-pro"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4.5"
    CLAUDE_OPUS_4_5 = "claude-opus-4.5"
    CLAUDE_OPUS_4_6 = "claude-opus-4.6"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4.5"
    QWEN_3_7_PLUS = "qwen3.7-plus"

    __str__ = str.__str__  # f"{ModelKey.GPT_5_5}" -> "gpt-5.5", not the repr


def _find_models_path() -> str:
    """Return path to models.json, for dev or installed package.

    Returns:
        Absolute path to models.json.

    Raises:
        FileNotFoundError: If models.json cannot be found.
    """
    import sys

    # Dev: models.json next to this module (always wins for local edits)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "models.json")
    if os.path.exists(candidate):
        return candidate

    # Installed: models.json is in share/pdf-to-md/ (data-files)
    installed = os.path.join(sys.prefix, "share", "pdf-to-md", "models.json")
    if os.path.exists(installed):
        return installed

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

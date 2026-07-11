"""
Local, offline OCR via Ollama and the Ollama-OCR project.

This module integrates https://github.com/imanoop7/Ollama-OCR as an optional
add-on so pages can be transcribed entirely on-device (no external API calls,
no per-token cost). It exposes a ``LocalOllamaProvider`` that plugs into the
existing ``VisionExtractor`` pipeline exactly like the cloud providers in
``llm_providers``, so the ``--local`` flag reuses all of the parallel
page-processing, prior-text, and markdown-assembly machinery unchanged.

Ollama-OCR (the ``ollama-ocr`` PyPI package) wraps an Ollama server and a
vision model (LLaVA, Llama 3.2 Vision, Granite Vision, MiniCPM-V, ...). It
runs as a background engine: images are sent to a locally running Ollama
server, which must have the requested model pulled. The helpers below detect
whether the server is reachable and pull models on demand while streaming
progress to the console (the first run of a model can take a while because the
weights are downloaded).
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from typing import List, Optional

from llm_providers import BaseProvider, TokenUsage, VisionResult

# Default Ollama vision model and server endpoint. The generate URL is the one
# Ollama-OCR's OCRProcessor expects; pull/tags URLs are derived from it.
DEFAULT_LOCAL_MODEL = "llama3.2-vision:11b"
DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


def _server_root(generate_url: str) -> str:
    """Return the Ollama server root (scheme://host:port) from a generate URL.

    Args:
        generate_url: Full Ollama generate endpoint, e.g.
            ``http://localhost:11434/api/generate``.

    Returns:
        Server root without the ``/api/...`` suffix.
    """
    return generate_url.split("/api/", 1)[0].rstrip("/")


def _pull_url(generate_url: str) -> str:
    """Return the Ollama ``/api/pull`` endpoint derived from a generate URL."""
    return f"{_server_root(generate_url)}/api/pull"


def _tags_url(generate_url: str) -> str:
    """Return the Ollama ``/api/tags`` endpoint derived from a generate URL."""
    return f"{_server_root(generate_url)}/api/tags"


def is_ollama_ocr_available() -> bool:
    """Return True when the optional ``ollama-ocr`` package is importable."""
    try:
        import ollama_ocr  # noqa: F401  pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - import failure path
        return False
    return True


def is_ollama_running(
    generate_url: str = DEFAULT_OLLAMA_GENERATE_URL,
    timeout: float = 3.0,
) -> bool:
    """Return True when an Ollama server responds at the given endpoint.

    Args:
        generate_url: Ollama generate endpoint whose host is probed.
        timeout: Socket timeout in seconds.

    Returns:
        True if the server's tags endpoint returns HTTP 200.
    """
    import requests  # pylint: disable=import-outside-toplevel

    try:
        response = requests.get(_tags_url(generate_url), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def list_installed_ollama_models(
    generate_url: str = DEFAULT_OLLAMA_GENERATE_URL,
) -> List[str]:
    """Return the names of models already pulled on the Ollama server.

    Args:
        generate_url: Ollama generate endpoint whose host is queried.

    Returns:
        List of installed model tags (empty if the server is unreachable).
    """
    import requests  # pylint: disable=import-outside-toplevel

    try:
        response = requests.get(_tags_url(generate_url), timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]


def ensure_ollama_model(
    model: str,
    generate_url: str = DEFAULT_OLLAMA_GENERATE_URL,
    verbose: bool = True,
) -> None:
    """Ensure a model is available on the Ollama server, pulling if needed.

    Streams pull progress to stdout so long downloads are visible. If the
    model is already installed this returns quickly without downloading.

    Args:
        model: Ollama model tag, e.g. ``llama3.2-vision:11b``.
        generate_url: Ollama generate endpoint (host is reused for pull/tags).
        verbose: Print progress messages while pulling.

    Raises:
        RuntimeError: If the Ollama server is unreachable or the pull fails.
    """
    import requests  # pylint: disable=import-outside-toplevel

    if not is_ollama_running(generate_url):
        raise RuntimeError(
            "Could not reach an Ollama server at "
            f"{_server_root(generate_url)}. Install Ollama from "
            "https://ollama.com and start it with `ollama serve`, or point "
            "--ollama-url at a running server."
        )

    installed = list_installed_ollama_models(generate_url)
    # Ollama tags include an implicit ":latest"; match with and without it.
    candidates = {model, f"{model}:latest", model.replace(":latest", "")}
    if any(name in candidates for name in installed):
        if verbose:
            print(f"[local] Model '{model}' already installed; skipping pull.")
        return

    if verbose:
        print(
            f"[local] Model '{model}' not found locally. Pulling from the "
            "Ollama registry (this can take several minutes on first run)..."
        )

    try:
        response = requests.post(
            _pull_url(generate_url),
            json={"name": model, "stream": True},
            stream=True,
            timeout=None,
        )
        response.raise_for_status()
    except Exception as error:
        raise RuntimeError(
            f"Failed to start pull of Ollama model '{model}': {error}"
        ) from error

    last_percent = -1
    for line in response.iter_lines():
        if not line:
            continue
        try:
            update = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue

        if update.get("error"):
            raise RuntimeError(
                f"Ollama failed to pull model '{model}': {update['error']}"
            )

        status = update.get("status", "")
        completed = update.get("completed")
        total = update.get("total")
        if verbose and total:
            percent = int(completed / total * 100) if completed else 0
            # Only print on 5% steps to avoid flooding the console.
            if percent >= last_percent + 5 or percent == 100:
                last_percent = percent
                mib = 1024 * 1024
                print(
                    f"[local] pull {model}: {status} "
                    f"{percent:3d}% "
                    f"({(completed or 0) / mib:.0f}/{total / mib:.0f} MiB)"
                )
        elif verbose and status:
            print(f"[local] pull {model}: {status}")

    if verbose:
        print(f"[local] Model '{model}' is ready.")


class LocalOllamaProvider(BaseProvider):
    """Vision provider backed by Ollama-OCR running against a local Ollama server.

    Implements the same ``process_vision`` contract as the cloud providers so
    it drops straight into ``VisionExtractor``. Each call writes the page image
    to a temporary file and hands it to Ollama-OCR's ``OCRProcessor``, which
    performs the on-device transcription. Token usage and cost are reported as
    zero because local inference is free.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        base_url: str = DEFAULT_OLLAMA_GENERATE_URL,
        language: str = "en",
    ):
        """Initialize the local provider.

        Args:
            model_name: Ollama vision model tag to run.
            base_url: Ollama generate endpoint used by Ollama-OCR.
            language: Language hint passed through to Ollama-OCR.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.language = language
        self._processor = None
        self._processor_lock = threading.Lock()

    def _get_processor(self):
        """Lazily construct and cache the Ollama-OCR ``OCRProcessor``.

        Returns:
            An ``OCRProcessor`` instance bound to this provider's model/URL.

        Raises:
            ImportError: If the optional ``ollama-ocr`` package is missing.
        """
        if self._processor is None:
            with self._processor_lock:
                if self._processor is None:
                    try:
                        from ollama_ocr import (  # pylint: disable=import-outside-toplevel
                            OCRProcessor,
                        )
                    except Exception as error:  # pragma: no cover
                        raise ImportError(
                            "The 'ollama-ocr' package is required for --local. "
                            "Install it with `pip install \".[local]\"` or "
                            "`pip install ollama-ocr`. See "
                            "https://github.com/imanoop7/Ollama-OCR"
                        ) from error
                    self._processor = OCRProcessor(
                        model_name=self.model_name,
                        base_url=self.base_url,
                    )
        return self._processor

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """Transcribe a page image with the local Ollama-OCR engine.

        Args:
            image_base64: Base64-encoded JPEG of the page.
            prompt: Instruction prompt (reused as Ollama-OCR ``custom_prompt``).
            prior_text: Optional first-pass digital text for context.
            model: Ignored; the model is fixed at construction time.
            max_tokens: Ignored; Ollama controls its own output length.

        Returns:
            VisionResult with the transcribed markdown and zero-cost usage.
        """
        full_prompt = prompt
        if prior_text:
            full_prompt = (
                f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"
            )

        processor = self._get_processor()

        # Ollama-OCR operates on file paths, so materialize the image briefly.
        image_bytes = base64.b64decode(image_base64)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            text = processor.process_image(
                image_path=tmp_path,
                format_type="markdown",
                preprocess=False,
                custom_prompt=full_prompt,
                language=self.language,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return VisionResult(
            text=text or "",
            usage=TokenUsage(input_tokens=0, output_tokens=0, cost_usd=0.0),
        )

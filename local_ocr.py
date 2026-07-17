"""
Local, offline OCR against a locally running Ollama server.

Pages are transcribed entirely on-device: no external API calls, no per-token
cost, and documents never leave the machine. ``LocalOllamaProvider`` plugs into
the existing ``VisionExtractor`` pipeline exactly like the cloud providers in
``llm_providers``, so ``--local`` reuses all of the parallel page-processing,
prior-text, and markdown-assembly machinery unchanged.

Ollama must be installed and running with a vision model pulled (Qwen-VL,
LLaVA, Granite Vision, MiniCPM-V, ...); the helpers below detect whether the
server is reachable and pull models on demand, streaming progress to the
console since the first run of a model downloads its weights.

This talks to Ollama's HTTP API directly rather than through a wrapper library,
which is what lets it size the context window per request (see
``DEFAULT_NUM_CTX``), bound each request with a timeout, and surface server
errors as exceptions instead of as page text.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from llm_providers import BaseProvider, TokenUsage, VisionResult

# Default Ollama vision model and server endpoint. Pull/tags URLs are derived
# from the generate URL.
DEFAULT_LOCAL_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Context window per request. We send exactly one page at a time: roughly
# ~1.2k image tokens, ~0.4k of prior_text, and ~0.3k of prompt (~2k in), plus
# ~1k generated back. 16k leaves several times that as headroom while staying
# far below the context some vision models default to -- qwen3-vl defaults to
# 262k, whose KV cache alone reserves tens of GB and pushes a 64GB machine into
# swap. Raise this only for pages that genuinely overflow it.
DEFAULT_NUM_CTX = 16384

# Output budget per page. Reasoning models (qwen3-vl and friends) spend this
# same budget on their chain of thought before answering, and thinking cannot
# reliably be turned off -- `think: false` and a `/no_think` prompt are both
# ignored. At 4096 they exhaust the budget mid-thought and return an empty
# page; 8192 leaves room to think *and* transcribe. Observed generation settles
# around 4.3-4.6k tokens even when given more, so this is headroom, not a
# target, and non-reasoning models stop well short of it and pay nothing.
DEFAULT_NUM_PREDICT = 8192

# Per-request ceiling. A large page on a small GPU is slow but not unbounded;
# without a timeout an unresponsive server would hang the run indefinitely.
DEFAULT_REQUEST_TIMEOUT = 600.0

# Local models often wrap a whole page of markdown in a ```markdown fence, which
# would render the page as a literal code block. Matches a fence that opens the
# text (with an optional language tag) and closes it on the final line.
_WRAPPING_FENCE_RE = re.compile(
    r"\A```[a-zA-Z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?```[ \t]*\Z",
    re.DOTALL,
)


def strip_wrapping_code_fence(text: str) -> str:
    """Unwrap page text that a model enclosed in a single fenced code block.

    Only an outer fence that spans the entire page is removed, so genuine code
    blocks inside a page are left untouched.

    Args:
        text: Raw page text returned by the model.

    Returns:
        The text with a whole-page wrapping fence removed, otherwise unchanged.
    """
    match = _WRAPPING_FENCE_RE.match(text.strip())
    return match.group("body").strip() if match else text


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
        model: Ollama model tag, e.g. ``qwen2.5vl:7b``.
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
    """Vision provider that talks to a local Ollama server over its HTTP API.

    Implements the same ``process_vision`` contract as the cloud providers so
    it drops straight into ``VisionExtractor``. Cost is always zero because
    inference is local, but token counts are reported from what Ollama returns.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        base_url: str = DEFAULT_OLLAMA_GENERATE_URL,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        """Initialize the local provider.

        Args:
            model_name: Ollama vision model tag to run.
            base_url: Ollama generate endpoint to post to.
            num_ctx: Context window in tokens. Sized for a single page; see
                DEFAULT_NUM_CTX for why the model default is not used.
            num_predict: Output token budget per page. Supersedes the caller's
                ``max_tokens``; see DEFAULT_NUM_PREDICT for why reasoning
                models need their own budget.
            timeout: Per-request timeout in seconds. A local page can take a
                while, but an unbounded wait would hang the run outright.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """Transcribe a page image with a local Ollama vision model.

        Args:
            image_base64: Base64-encoded JPEG of the page.
            prompt: Instruction prompt for the model.
            prior_text: Optional first-pass digital text for context.
            model: Ignored; the model is fixed at construction time.
            max_tokens: Floor for the output budget. The provider's own
                ``num_predict`` wins when it is larger, since a reasoning model
                needs room to think before it transcribes.

        Returns:
            VisionResult with the transcribed markdown and zero-cost usage,
            carrying the token counts Ollama reported.

        Raises:
            RuntimeError: If the server is unreachable, returns an error, or
                replies with an unusable body, so the caller's retry and
                failed-page tracking engage rather than a bad page being
                written into the document.
        """
        import requests  # pylint: disable=import-outside-toplevel

        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max(self.num_predict, max_tokens),
            },
        }

        try:
            response = requests.post(
                self.base_url, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as error:
            raise RuntimeError(
                f"Ollama request failed for model '{self.model_name}' at "
                f"{self.base_url}: {error}"
            ) from error

        if data.get("error"):
            raise RuntimeError(
                f"Ollama returned an error for model '{self.model_name}': "
                f"{data['error']}"
            )

        text = data.get("response", "")

        # A reasoning model that spends its whole budget thinking returns
        # done_reason="length" with little or no answer. That would otherwise
        # land in the document as a silently blank page, so fail loudly and
        # name the flag that fixes it.
        if data.get("done_reason") == "length" and not text.strip():
            raise RuntimeError(
                f"Model '{self.model_name}' hit its output budget "
                f"({max(self.num_predict, max_tokens)} tokens) without "
                "returning any text. Reasoning models spend this budget "
                "thinking before they answer; raise --local-num-predict (and "
                "--local-num-ctx to fit it), or use a non-reasoning vision "
                "model such as qwen2.5vl:7b."
            )

        return VisionResult(
            text=strip_wrapping_code_fence(text) if text else "",
            usage=TokenUsage(
                input_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
                cost_usd=0.0,
            ),
        )

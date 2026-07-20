"""
Local, offline OCR against a locally running vLLM OpenAI-compatible server.

This is the second local backend (alongside ``local_ocr``'s Ollama path). It
targets purpose-built document-parsing VLMs that ship on Hugging Face and are
served with vLLM rather than Ollama -- in particular
`KDLAI/KDL-Frontier-Parser-nano <https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano>`_,
a 1.2B Qwen2-VL-based parser (MinerU 2.5 lineage) tuned for OCR, tables, and
charts. Like the Ollama path it plugs straight into ``VisionExtractor``, so
``--local --local-backend vllm`` reuses all of the parallel page-processing,
prior-text, and markdown-assembly machinery unchanged.

Why a separate server rather than in-process weights: KDL-Frontier-Parser and
its siblings need ``--trust-remote-code`` and run best under vLLM's paged-KV
engine. Offloading inference to a ``vllm serve`` process (exactly as the Ollama
path offloads to ``ollama serve``) keeps this package free of a heavy
torch/transformers/vLLM dependency and lets the server own GPU memory and
batching. Start one like this::

    vllm serve KDLAI/KDL-Frontier-Parser-nano \\
      --served-model-name kdl-frontier-parser-nano \\
      --max-model-len 8192 \\
      --gpu-memory-utilization 0.85 \\
      --max-num-seqs 24 \\
      --trust-remote-code \\
      --limit-mm-per-prompt '{"image":1}'

Inference is a single end-to-end pass per page: the page image plus the shared
transcription prompt go to the server's ``/v1/chat/completions`` endpoint. The
model card asks for greedy decoding (``temperature=0``), ``enable_thinking``
off in the chat template, and ``skip_special_tokens`` off when decoding; all
three are set on every request below.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from llm_providers import BaseProvider, TokenUsage, VisionResult
from local_ocr import strip_wrapping_code_fence

# Default served-model name (matches the --served-model-name in the model
# card's vllm serve command) and the OpenAI-compatible base URL vLLM listens on.
DEFAULT_VLLM_MODEL = "kdl-frontier-parser-nano"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

# Output budget per page. The model card serves with --max-model-len 8192, and a
# page image plus prior text and prompt already consume a few thousand tokens of
# that window, so the answer must fit in what remains. 4096 comfortably covers a
# dense page of markdown while staying clear of the context limit.
DEFAULT_MAX_TOKENS = 4096

# Per-request ceiling. A large page on a small GPU is slow but not unbounded;
# without a timeout an unresponsive server would hang the run indefinitely.
DEFAULT_REQUEST_TIMEOUT = 600.0

# ---------------------------------------------------------------------------
# Baidu Unlimited-OCR (DeepSeek-OCR lineage) defaults.
#
# Unlimited-OCR is a separate vLLM-served model. It is a distinct model per
# server, so it defaults to its own port (8001) rather than sharing KDL's 8000
# -- on one GPU you run KDL and Unlimited-OCR as two servers. Its 32k context
# fits long pages, so its per-page output budget is larger than KDL's.
DEFAULT_UNLIMITED_MODEL = "unlimited-ocr"
DEFAULT_UNLIMITED_BASE_URL = "http://localhost:8001/v1"
DEFAULT_UNLIMITED_MAX_TOKENS = 8192

# Unlimited-OCR has no chat template and is trained for a fixed recipe: the
# prompt must begin with a literal <image> placeholder, and <|grounding|> asks
# for a document-to-markdown pass. A generic "write markdown" instruction (or a
# missing <image>) makes the model return empty output, so this prompt is used
# verbatim and the caller's prompt is ignored.
DEFAULT_UNLIMITED_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


# Grounding markup emitted by DeepSeek-OCR-lineage models: <|ref|>text<|/ref|>
# wraps a span of content and <|det|>[[x,y,...]]<|/det|> carries its pixel box.
_DET_BLOCK_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_REF_WRAP_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_LEFTOVER_GROUNDING_RE = re.compile(r"<\|/?(?:ref|det|grounding)\|>")


def strip_grounding_tokens(text: str) -> str:
    """Turn Unlimited-OCR's grounded output into clean markdown.

    DeepSeek-OCR-lineage models interleave the transcription with grounding
    markup: ``<|ref|>span<|/ref|>`` wraps a piece of content and
    ``<|det|>[[x1,y1,x2,y2]]<|/det|>`` gives its bounding box. The boxes are
    coordinates, not document content, so they are dropped; the ``<|ref|>``
    wrappers are unwrapped to keep the text they enclose.

    Args:
        text: Raw model output, possibly carrying grounding markup.

    Returns:
        The text with coordinate boxes removed, reference spans unwrapped, and
        any stray grounding markers cleaned up.
    """
    # Drop coordinate boxes first so their contents can't survive as stray text.
    text = _DET_BLOCK_RE.sub("", text)
    # Unwrap reference spans, keeping the enclosed content.
    text = _REF_WRAP_RE.sub(r"\1", text)
    # Remove the grounding marker and any unpaired ref/det tags left behind.
    text = text.replace("<|grounding|>", "")
    text = _LEFTOVER_GROUNDING_RE.sub("", text)
    return text.strip()


def _models_url(base_url: str) -> str:
    """Return the ``/models`` endpoint derived from an OpenAI-compatible base.

    Args:
        base_url: vLLM OpenAI base URL, e.g. ``http://localhost:8000/v1``.

    Returns:
        The ``.../models`` listing endpoint used to probe the server.
    """
    return f"{base_url.rstrip('/')}/models"


def is_vllm_running(
    base_url: str = DEFAULT_VLLM_BASE_URL,
    timeout: float = 3.0,
) -> bool:
    """Return True when a vLLM OpenAI-compatible server responds.

    Args:
        base_url: vLLM OpenAI base URL whose ``/models`` endpoint is probed.
        timeout: Socket timeout in seconds.

    Returns:
        True if the server's models endpoint returns HTTP 200.
    """
    import requests  # pylint: disable=import-outside-toplevel

    try:
        response = requests.get(_models_url(base_url), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def list_served_vllm_models(
    base_url: str = DEFAULT_VLLM_BASE_URL,
) -> List[str]:
    """Return the model ids the vLLM server is currently serving.

    Args:
        base_url: vLLM OpenAI base URL whose ``/models`` endpoint is queried.

    Returns:
        List of served model ids (empty if the server is unreachable).
    """
    import requests  # pylint: disable=import-outside-toplevel

    try:
        response = requests.get(_models_url(base_url), timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    return [m.get("id", "") for m in payload.get("data", []) if m.get("id")]


def ensure_vllm_server(
    model: str,
    base_url: str = DEFAULT_VLLM_BASE_URL,
    verbose: bool = True,
    url_flag: str = "--vllm-url",
    model_flag: str = "--vllm-model",
) -> None:
    """Verify a vLLM server is reachable and serving the requested model.

    Unlike the Ollama path this never downloads or launches anything: a vLLM
    server is started out of band (``vllm serve ...``) and owns its own model
    loading. This only checks that such a server is up and that ``model`` is one
    of the ids it advertises, failing with an actionable message otherwise.

    Args:
        model: Served-model name expected on the server.
        base_url: vLLM OpenAI base URL.
        verbose: Print a confirmation line when the model is found.
        url_flag: CLI flag named in the "unreachable" message.
        model_flag: CLI flag named in the "wrong model" message.

    Raises:
        RuntimeError: If the server is unreachable or is not serving ``model``.
    """
    if not is_vllm_running(base_url):
        raise RuntimeError(
            f"Could not reach a vLLM server at {base_url}. Start one with "
            f"`vllm serve <model-repo> --served-model-name {model}` (see the "
            f"README for the full command), or point {url_flag} at a running "
            "server."
        )

    served = list_served_vllm_models(base_url)
    if served and model not in served:
        raise RuntimeError(
            f"vLLM server at {base_url} is running but does not serve a model "
            f"named '{model}'. It is serving: {', '.join(served)}. Pass the "
            f"right name with {model_flag}, or restart the server with "
            f"--served-model-name {model}."
        )

    if verbose:
        print(f"[vllm] Server at {base_url} is serving '{model}'.")


class VllmOpenAIProvider(BaseProvider):
    """Vision provider backed by a local vLLM OpenAI-compatible server.

    Implements the same ``process_vision`` contract as the cloud providers so it
    drops straight into ``VisionExtractor``. Cost is always zero because
    inference is local, but token counts are reported from what vLLM returns.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VLLM_MODEL,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        """Initialize the local vLLM provider.

        Args:
            model_name: Served-model name to request (the vLLM
                ``--served-model-name``).
            base_url: vLLM OpenAI-compatible base URL, e.g.
                ``http://localhost:8000/v1``.
            max_tokens: Output token budget per page. Caps the model's answer;
                see DEFAULT_MAX_TOKENS for why it stays below the context limit.
            timeout: Per-request timeout in seconds. A local page can take a
                while, but an unbounded wait would hang the run outright.
        """
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        # A local server needs no real key, but the OpenAI client requires a
        # non-empty one; vLLM ignores its value unless started with --api-key.
        self.client = OpenAI(
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=timeout,
        )

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """Transcribe a page image with a local vLLM document-parser model.

        Args:
            image_base64: Base64-encoded JPEG of the page.
            prompt: Instruction prompt for the model.
            prior_text: Optional first-pass digital text for context.
            model: Ignored; the served model is fixed at construction time.
            max_tokens: Floor for the output budget. The provider's own
                ``max_tokens`` wins when it is larger.

        Returns:
            VisionResult with the transcribed markdown and zero-cost usage,
            carrying the token counts vLLM reported.

        Raises:
            RuntimeError: If the server is unreachable, returns an error, or
                replies with an unusable body, so the caller's retry and
                failed-page tracking engage rather than a bad page being
                written into the document.
        """
        full_prompt = self._build_prompt(prompt, prior_text)
        budget = max(self.max_tokens, max_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        f"{image_base64}"
                                    )
                                },
                            },
                        ],
                    }
                ],
                # The model card asks for greedy decoding.
                temperature=0.0,
                max_tokens=budget,
                extra_body=self._extra_body(),
            )
        except Exception as error:
            raise RuntimeError(
                f"vLLM request failed for model '{self.model_name}' at "
                f"{self.base_url}: {error}"
            ) from error

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""

        # A truncated answer with no usable text would otherwise land in the
        # document as a silently blank page, so fail loudly and name the flag.
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        if finish_reason == "length" and not text.strip():
            raise RuntimeError(
                f"Model '{self.model_name}' hit its output budget "
                f"({budget} tokens) without returning any text. Raise "
                f"{self._max_tokens_flag} (and the server's --max-model-len "
                "to fit it)."
            )

        usage = response.usage
        return VisionResult(
            text=self._postprocess(text) if text else "",
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cost_usd=0.0,
            ),
        )

    # CLI flag named in the budget-exhaustion error; subclasses override it.
    _max_tokens_flag = "--vllm-max-tokens"

    def _build_prompt(self, prompt: str, prior_text: Optional[str]) -> str:
        """Return the text prompt sent alongside the page image.

        The default keeps the caller's prompt and appends any first-pass
        digital text in ``<prior_text>`` tags, exactly like the cloud
        providers. Subclasses whose model needs a fixed prompt recipe override
        this.

        Args:
            prompt: Instruction prompt supplied by the extractor.
            prior_text: Optional first-pass digital text for context.

        Returns:
            The full text prompt for the request.
        """
        if prior_text:
            return f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"
        return prompt

    def _extra_body(self) -> dict:
        """Return vLLM-specific request options merged into the JSON body.

        The default keeps the parser's special tokens in the decoded output and
        disables the chat template's thinking block, both per the KDL model
        card. Subclasses override this when their model needs different options.

        Returns:
            A dict passed as the OpenAI client's ``extra_body``.
        """
        return {
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _postprocess(self, text: str) -> str:
        """Clean the raw model output into page markdown.

        The default only unwraps a whole-page code fence. Subclasses whose
        model emits grounding or other markup override this.

        Args:
            text: Raw text returned by the model.

        Returns:
            Cleaned page markdown.
        """
        return strip_wrapping_code_fence(text)


class UnlimitedOcrProvider(VllmOpenAIProvider):
    """Vision provider for Baidu Unlimited-OCR served on a local vLLM server.

    Unlimited-OCR is DeepSeek-OCR lineage and shares the same vLLM
    OpenAI-compatible transport as :class:`VllmOpenAIProvider`, but its recipe
    differs in three ways handled here:

    - It has no chat template and is trained on a fixed prompt that must begin
      with ``<image>``. The extractor's generic markdown prompt (and any prior
      text) is therefore ignored in favour of ``ocr_prompt``.
    - Its output interleaves ``<|ref|>`` / ``<|det|>`` grounding markup, which
      is stripped back to plain markdown.
    - It needs no ``enable_thinking`` chat-template kwarg (there is no
      template), only ``skip_special_tokens=False`` so the grounding markup
      survives long enough to be stripped.

    The server must be started with the model's no-repeat-ngram logits
    processor (see the README); without it long pages loop on coordinate
    tokens. That is a server-launch concern, not a client one.
    """

    _max_tokens_flag = "--unlimited-max-tokens"

    def __init__(
        self,
        model_name: str = DEFAULT_UNLIMITED_MODEL,
        base_url: str = DEFAULT_UNLIMITED_BASE_URL,
        max_tokens: int = DEFAULT_UNLIMITED_MAX_TOKENS,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        ocr_prompt: str = DEFAULT_UNLIMITED_OCR_PROMPT,
    ):
        """Initialize the Unlimited-OCR provider.

        Args:
            model_name: Served-model name to request.
            base_url: vLLM OpenAI-compatible base URL.
            max_tokens: Output token budget per page.
            timeout: Per-request timeout in seconds.
            ocr_prompt: Fixed recipe prompt sent for every page; must begin
                with ``<image>``.
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.ocr_prompt = ocr_prompt

    def _build_prompt(self, prompt: str, prior_text: Optional[str]) -> str:
        """Return the fixed recipe prompt, ignoring the caller's prompt.

        Unlimited-OCR returns empty output unless the prompt follows its recipe
        (a leading ``<image>`` and its grounding instruction), so the
        extractor's generic prompt and prior text are deliberately dropped.

        Args:
            prompt: Ignored.
            prior_text: Ignored.

        Returns:
            The provider's fixed OCR prompt.
        """
        return self.ocr_prompt

    def _extra_body(self) -> dict:
        """Keep special tokens so grounding markup can be stripped afterwards.

        Unlike KDL there is no chat template, so no ``chat_template_kwargs`` is
        sent -- only ``skip_special_tokens=False``.

        Returns:
            A dict passed as the OpenAI client's ``extra_body``.
        """
        return {"skip_special_tokens": False}

    def _postprocess(self, text: str) -> str:
        """Strip grounding markup, then unwrap any whole-page code fence.

        Args:
            text: Raw model output with ``<|ref|>`` / ``<|det|>`` markup.

        Returns:
            Clean page markdown.
        """
        return strip_wrapping_code_fence(strip_grounding_tokens(text))

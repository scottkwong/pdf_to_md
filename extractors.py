"""
PDF extraction strategy abstraction layer.

Provides a unified interface for different PDF-to-markdown extraction methods:
- Vision-based extraction using LLM providers (existing functionality)
- LlamaParse API extraction (new functionality)

This module can be extended to support additional extraction methods.
"""
import base64
import io
import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from pdf2image import convert_from_path
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)
from tqdm import tqdm
from digital_text_parsers import (
    BaseDigitalTextParser,
    create_digital_text_parser,
)

if TYPE_CHECKING:
    from llm_providers import BaseProvider, VisionResult


# Retry envelopes. A provider quota resets on a minute scale, so a rate-limited
# page needs a retry window longer than that window to survive; a genuine error
# (bad request, decode failure) should still fail fast rather than stall a run
# for five minutes. The two policies below are selected per-exception.
_TRANSIENT_WAIT = wait_random_exponential(min=1.0 / 5000, max=5)
_TRANSIENT_STOP = stop_after_attempt(3)
_RATE_LIMIT_WAIT = wait_random_exponential(multiplier=2, max=60)
_RATE_LIMIT_STOP = stop_after_delay(300) | stop_after_attempt(12)

# Concurrency for the end-of-run recovery pass. Deliberately far below the
# default fan-out: the pages being retried are the ones a saturated quota
# rejected, so the retry has to ask for less, not the same again.
RECOVERY_CONCURRENCY = 3


def is_rate_limit_error(exc: BaseException) -> bool:
    """
    Report whether an exception is a provider rate-limit (HTTP 429) rejection.

    Detection is duck-typed rather than keyed to one SDK's exception class:
    openai, anthropic, and google-genai all surface 429s differently, and the
    Fireworks and OpenRouter providers reuse the openai client against other
    hosts.

    Args:
        exc: Exception raised by a provider call.

    Returns:
        True if the exception represents a rate-limit rejection.
    """
    if type(exc).__name__ == "RateLimitError":
        return True

    response = getattr(exc, "response", None)
    candidates = (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(response, "status_code", None),
    )
    return any(code == 429 for code in candidates)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """
    Read the provider's Retry-After hint, in seconds, if it sent one.

    Args:
        exc: Exception raised by a provider call.

    Returns:
        Seconds to wait, or None when absent or not a plain number. HTTP-date
        forms are ignored: these APIs send deltas, and guessing at a date parse
        risks a wait far longer than the quota window.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None

    try:
        raw = headers.get("retry-after")
    except AttributeError:
        return None

    if raw is None:
        return None

    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None

    # Ignore nonsense (negative, or a wait longer than the whole envelope).
    return seconds if 0 <= seconds <= 300 else None


# Transport and server-side faults worth a second attempt. Matched by class
# name because each SDK defines its own hierarchy.
_TRANSIENT_ERROR_NAMES = frozenset({
    "APIConnectionError",
    "APITimeoutError",
    "ConnectionError",
    "InternalServerError",
    "ReadTimeout",
    "ServiceUnavailableError",
    "Timeout",
})


def is_retryable_error(exc: BaseException) -> bool:
    """
    Report whether re-sending a page could plausibly succeed.

    Throttling and transport faults are worth another attempt. A deterministic
    fault (malformed request, auth failure, a bug in our own response handling)
    will fail identically the second time, so retrying it only doubles the cost
    and delay of a run that is already going to fail.

    Args:
        exc: Exception raised while processing a page.

    Returns:
        True if the failure is transient.
    """
    if is_rate_limit_error(exc):
        return True

    if type(exc).__name__ in _TRANSIENT_ERROR_NAMES:
        return True

    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and status >= 500


@dataclass
class PageFailure:
    """A page that failed to process, and whether a retry could help."""

    index: int
    message: str
    retryable: bool


def _provider_safe_concurrency(provider: object, default: int = 10) -> int:
    """
    Read a provider's declared safe page concurrency.

    Args:
        provider: Provider instance, which may predate MAX_SAFE_CONCURRENCY or
            be a test double that answers any attribute.
        default: Fallback when the provider declares nothing usable.

    Returns:
        A positive page-concurrency limit.
    """
    declared = getattr(provider, "MAX_SAFE_CONCURRENCY", default)
    if isinstance(declared, bool) or not isinstance(declared, int):
        return default
    return declared if declared >= 1 else default


def _wait_for_exception(retry_state) -> float:
    """Pick the backoff envelope that matches the failure, in seconds."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None and is_rate_limit_error(exc):
        hinted = retry_after_seconds(exc) or 0.0
        return max(hinted, _RATE_LIMIT_WAIT(retry_state))
    return _TRANSIENT_WAIT(retry_state)


def _stop_for_exception(retry_state) -> bool:
    """Stop rate-limited calls on the long envelope, others on the short one."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None and is_rate_limit_error(exc):
        return _RATE_LIMIT_STOP(retry_state)
    return _TRANSIENT_STOP(retry_state)


@dataclass
class ExtractionResult:
    """Result from PDF extraction."""

    markdown: str
    page_count: int
    metadata: Optional[dict] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base class for PDF extraction strategies."""

    @abstractmethod
    def extract(
        self,
        pdf_path: str,
        output_dir: str,
        verbose: bool = True,
    ) -> ExtractionResult:
        """
        Extract markdown from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for any intermediate files (e.g., cached images).
            verbose: Whether to print progress.

        Returns:
            ExtractionResult containing the markdown and metadata.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the extractor name for display purposes."""
        pass


class VisionExtractor(BaseExtractor):
    """
    Vision-based extraction using LLM providers.

    Converts PDF pages to images and processes each with a vision-capable LLM.
    This wraps the existing page-by-page vision processing workflow.
    """

    def __init__(
        self,
        provider: "BaseProvider",
        model_id: str,
        mode: str = "vt",
        max_parallel_pages: Optional[int] = None,
        digital_text_parser: str = "auto",
    ):
        """
        Initialize VisionExtractor.

        Args:
            provider: LLM provider instance (from llm_providers module).
            model_id: Model identifier for the provider.
            mode: Processing mode - 'v' for vision-only, 'vt' for vision-and-text.
            max_parallel_pages: Maximum number of pages to process in parallel.
                None resolves to the provider's MAX_SAFE_CONCURRENCY, so a
                provider with a tight quota is not fanned out into throttling.
            digital_text_parser: Parser engine used for first-pass digital text
                parsing in 'vt' mode. Valid values are auto, pypdf, pymupdf.
        """
        self.provider = provider
        self.model_id = model_id
        self.mode = mode
        self.max_parallel_pages = (
            max_parallel_pages
            if max_parallel_pages is not None
            else _provider_safe_concurrency(provider)
        )
        self._digital_text_parser_selection = create_digital_text_parser(
            digital_text_parser
        )
        self.digital_text_parser: BaseDigitalTextParser = (
            self._digital_text_parser_selection.parser
        )
        self._cost_lock = threading.Lock()
        self._page_costs: list[dict] = []
        self._pricing: tuple[float, float] | None = None

        # Validate mode
        if mode not in ["v", "vt"]:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are ['v', 'vt'].")

    @property
    def name(self) -> str:
        return f"Vision ({self.model_id})"

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_cost_per_mtok, output_cost_per_mtok) from models.json, cached."""
        if self._pricing is None:
            from llm_providers import load_models_config
            models_config = load_models_config()
            # Find config matching our model_id (check both name keys and direct_id)
            for _name, cfg in models_config.items():
                if self.model_id in (cfg.get("direct_id"), cfg.get("openrouter_id"), _name):
                    self._pricing = (
                        cfg.get("input_cost_per_mtok", 0.0),
                        cfg.get("output_cost_per_mtok", 0.0),
                    )
                    return self._pricing
            self._pricing = (0.0, 0.0)
        return self._pricing

    def extract(
        self,
        pdf_path: str,
        output_dir: str,
        verbose: bool = True,
    ) -> ExtractionResult:
        """
        Extract markdown from PDF using vision LLM.

        Pages are processed in parallel up to max_parallel_pages concurrent
        requests. Results are reconstructed in correct page order.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for cached images.
            verbose: Whether to print progress.

        Returns:
            ExtractionResult with markdown content.
        """
        self._page_costs = []
        pdf_file_name = os.path.basename(pdf_path)

        # Load page images and prior text
        images, prior_texts = self._load_page_data(pdf_path, output_dir)

        # Process all pages in parallel
        results, failed_pages = self._process_pages_parallel(
            images, prior_texts, pdf_file_name, verbose
        )

        # Give throttled pages a second chance before they become placeholders
        if failed_pages:
            results, failed_pages = self._recover_failed_pages(
                images, prior_texts, pdf_file_name, verbose, results, failed_pages
            )

        # Log any errors
        self._log_processing_errors(failed_pages, len(images), results)

        # Assemble final markdown in page order
        markdown_content = self._assemble_markdown(
            results, len(images), pdf_file_name
        )

        # Aggregate cost data
        total_input_tokens = sum(c["input_tokens"] for c in self._page_costs)
        total_output_tokens = sum(c["output_tokens"] for c in self._page_costs)
        total_cost_usd = sum(c["cost_usd"] for c in self._page_costs)

        return ExtractionResult(
            markdown="\n".join(markdown_content),
            page_count=len(images),
            metadata={
                "extractor": "vision",
                "model": self.model_id,
                "mode": self.mode,
                "max_parallel_pages": self.max_parallel_pages,
                "digital_text_parser_requested": (
                    self._digital_text_parser_selection.requested_parser
                ),
                "digital_text_parser_resolved": (
                    self._digital_text_parser_selection.resolved_parser
                ),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_usd": total_cost_usd,
            },
        )

    def _load_page_data(
        self, pdf_path: str, output_dir: str
    ) -> tuple[List[Image.Image], List[Optional[str]]]:
        """
        Load page images and extract prior text from PDF.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for cached images.

        Returns:
            Tuple of (images, prior_texts) lists.

        Raises:
            ValueError: If image and text counts don't match.
        """
        images = self._pdf_to_images_with_storage(pdf_path, output_dir)

        if self.mode == "v":
            prior_texts: List[Optional[str]] = [None] * len(images)
        else:  # mode == 'vt'
            prior_texts = self._get_prior_text(pdf_path)

        if len(prior_texts) != len(images):
            raise ValueError(
                f"The number of prior texts ({len(prior_texts)}) does not match "
                f"the number of images ({len(images)})."
            )

        return images, prior_texts

    def _process_pages_parallel(
        self,
        images: List[Image.Image],
        prior_texts: List[Optional[str]],
        pdf_file_name: str,
        verbose: bool,
        page_indices: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        desc: str = "Processing pages",
    ) -> tuple[dict[int, str], list[tuple[int, str]]]:
        """
        Process pages in parallel using ThreadPoolExecutor.

        Args:
            images: List of page images.
            prior_texts: List of prior text for each page.
            pdf_file_name: Name of the PDF file for headers.
            verbose: Whether to show progress bar.
            page_indices: Page indices to process. Defaults to every page; the
                recovery sweep passes just the pages that failed.
            max_workers: Concurrency for this pass. Defaults to the extractor's
                max_parallel_pages.
            desc: Progress bar label.

        Returns:
            Tuple of (results dict, failed_pages list).
            results: Maps page index to markdown content.
            failed_pages: List of (page_index, error_message) tuples.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_pages = len(images)
        indices = (
            list(range(total_pages)) if page_indices is None else list(page_indices)
        )
        workers = max(1, max_workers or self.max_parallel_pages)
        results: dict[int, str] = {}
        failed_pages: list[tuple[int, str]] = []

        logger.debug(
            f"Starting parallel extraction: {len(indices)} of {total_pages} "
            f"pages, max_workers={workers}"
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_page, ix, images[ix], prior_texts[ix],
                    pdf_file_name, total_pages
                ): ix
                for ix in indices
            }

            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(iterator, total=len(futures), desc=desc)

            for future in iterator:
                page_index = futures[future]
                try:
                    result_index, content = future.result()
                    results[result_index] = content
                    logger.debug(
                        f"Collected result for page {result_index + 1} "
                        f"(completed {len(results)}/{total_pages})"
                    )
                except Exception as e:
                    failed_pages.append(
                        PageFailure(page_index, str(e), is_retryable_error(e))
                    )
                    logger.error(
                        f"FAILED: Page {page_index + 1} failed to process: {e}"
                    )

        return results, failed_pages

    def _recover_failed_pages(
        self,
        images: List[Image.Image],
        prior_texts: List[Optional[str]],
        pdf_file_name: str,
        verbose: bool,
        results: dict[int, str],
        failed_pages: list[PageFailure],
    ) -> tuple[dict[int, str], list[PageFailure]]:
        """
        Re-run the transiently-failed pages, once, at reduced concurrency.

        A block of mid-run failures is usually provider throttling rather than
        bad pages: the worker pool saturates a per-minute quota and every
        in-flight request is rejected together. The quota window has normally
        reset by the time the first pass ends, so one narrow retry recovers
        pages that would otherwise be written out as [ERROR] placeholders.

        Args:
            images: List of page images.
            prior_texts: List of prior text for each page.
            pdf_file_name: Name of the PDF file for headers.
            verbose: Whether to show progress.
            results: Results collected so far, updated in place with recoveries.
            failed_pages: Pages that failed the first pass.

        Returns:
            Tuple of (results dict, still-failing pages).
        """
        retryable = [failure for failure in failed_pages if failure.retryable]
        permanent = [failure for failure in failed_pages if not failure.retryable]

        if not retryable:
            logger.info(
                f"No recovery pass: all {len(permanent)} failure(s) are "
                f"deterministic, so a retry would fail identically"
            )
            return results, failed_pages

        retry_indices = [failure.index for failure in retryable]
        retry_workers = max(1, min(RECOVERY_CONCURRENCY, self.max_parallel_pages))

        logger.info(
            f"Retrying {len(retry_indices)} transiently-failed page(s) at "
            f"max_workers={retry_workers}"
        )
        if verbose:
            print(
                f"\n{len(retry_indices)} page(s) hit throttling or a transport "
                f"error; retrying at {retry_workers}-way concurrency."
            )

        recovered, still_failed = self._process_pages_parallel(
            images,
            prior_texts,
            pdf_file_name,
            verbose,
            page_indices=retry_indices,
            max_workers=retry_workers,
            desc="Retrying failed pages",
        )

        results.update(recovered)
        logger.info(
            f"Recovery pass: {len(recovered)} recovered, "
            f"{len(still_failed)} still failing"
        )

        remaining = permanent + still_failed
        remaining.sort(key=lambda failure: failure.index)
        return results, remaining

    def _process_single_page(
        self,
        page_index: int,
        image: Image.Image,
        prior_text: Optional[str],
        pdf_file_name: str,
        total_pages: int,
    ) -> tuple[int, str]:
        """
        Process a single page and return its markdown content.

        Args:
            page_index: Zero-based page index.
            image: PIL Image of the page.
            prior_text: Optional extracted text for context.
            pdf_file_name: Name of the PDF file for header.
            total_pages: Total number of pages (for logging).

        Returns:
            Tuple of (page_index, markdown_content_with_header).
        """
        logger.debug(
            f"Starting page {page_index + 1}/{total_pages} "
            f"(0-indexed: {page_index})"
        )

        image_base64 = self._pdf_image_to_base64_str(image)
        vision_result = self._process_image_with_provider(image_base64, prior_text)

        # Calculate cost for this page
        usage = vision_result.usage
        if usage.cost_usd is not None:
            page_cost = usage.cost_usd
        else:
            input_rate, output_rate = self._get_pricing()
            page_cost = (
                usage.input_tokens * input_rate / 1_000_000
                + usage.output_tokens * output_rate / 1_000_000
            )

        with self._cost_lock:
            self._page_costs.append({
                "page": page_index + 1,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": page_cost,
            })

        logger.debug(
            f"Completed page {page_index + 1}/{total_pages} — "
            f"in={usage.input_tokens:,} out={usage.output_tokens:,} "
            f"cost=${page_cost:.4f}"
        )

        page_header = f"File: {pdf_file_name}; Page: {page_index + 1}\n"
        return page_index, page_header + vision_result.text

    def _log_processing_errors(
        self,
        failed_pages: list[PageFailure],
        total_pages: int,
        results: dict[int, str],
    ) -> None:
        """
        Log errors for failed or missing pages.

        Args:
            failed_pages: Pages that failed after any recovery pass.
            total_pages: Expected total number of pages.
            results: Dict of successfully processed pages.
        """
        if failed_pages:
            failed_nums = [failure.index + 1 for failure in failed_pages]
            logger.error(
                f"PAGE PROCESSING ERRORS: {len(failed_pages)} of {total_pages} "
                f"pages failed to process. Failed pages: {failed_nums}"
            )
            for failure in failed_pages:
                logger.error(f"  Page {failure.index + 1}: {failure.message}")

        expected_pages = set(range(total_pages))
        received_pages = set(results.keys())
        missing_pages = expected_pages - received_pages

        if missing_pages:
            missing_nums = sorted([p + 1 for p in missing_pages])
            logger.error(
                f"MISSING PAGES: Expected {total_pages} pages but only "
                f"received {len(results)}. Missing pages: {missing_nums}"
            )

    def _assemble_markdown(
        self,
        results: dict[int, str],
        total_pages: int,
        pdf_file_name: str,
    ) -> list[str]:
        """
        Assemble markdown content in correct page order.

        Args:
            results: Dict mapping page index to markdown content.
            total_pages: Total number of pages expected.
            pdf_file_name: Name of PDF file for error placeholders.

        Returns:
            List of markdown strings in page order.
        """
        assembly_order = list(range(total_pages))
        logger.debug(f"Assembly order (0-indexed): {assembly_order}")

        markdown_content = []
        for i in assembly_order:
            if i in results:
                markdown_content.append(results[i])
            else:
                placeholder = (
                    f"File: {pdf_file_name}; Page: {i + 1}\n"
                    f"[ERROR: Page {i + 1} failed to process]\n"
                )
                markdown_content.append(placeholder)

        # Verify assembly order
        assembled_indices = [i for i in assembly_order if i in results]
        if assembled_indices != sorted(assembled_indices):
            logger.error(
                f"PAGE ORDER ERROR: Pages were not assembled in sequential "
                f"order. Assembled indices: {assembled_indices}"
            )
        else:
            logger.debug(
                f"Assembled {len(markdown_content)} pages in order: "
                f"{[i + 1 for i in assembly_order]}"
            )

        return markdown_content

    def _pdf_to_images_with_storage(
        self, pdf_path: str, output_dir: str
    ) -> List[Image.Image]:
        """
        Load images from cache or convert PDF to images.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for cached images.

        Returns:
            List of PIL Image objects.
        """
        base_name = os.path.basename(pdf_path).rsplit(".", 1)[0]
        image_folder = os.path.join(output_dir, base_name + "_images")

        cached = self._load_cached_images(image_folder)
        if cached:
            return cached

        # Render into a staging directory and move it into place only after every
        # page is written. Rendering straight into image_folder leaves an empty or
        # partial cache behind if convert_from_path fails (missing poppler, corrupt
        # PDF, interrupt) -- and since the cache was keyed on the directory merely
        # existing, every later run would load zero pages and fail a page-count
        # mismatch until the directory was deleted by hand.
        staging_folder = image_folder + ".partial"
        shutil.rmtree(staging_folder, ignore_errors=True)
        os.makedirs(staging_folder)

        try:
            images = convert_from_path(pdf_path, thread_count=os.cpu_count())
            for i, image in enumerate(images):
                image.save(
                    os.path.join(staging_folder, f"{base_name}_image_{i}.png")
                )
        except BaseException:
            shutil.rmtree(staging_folder, ignore_errors=True)
            raise

        shutil.rmtree(image_folder, ignore_errors=True)
        os.rename(staging_folder, image_folder)

        return images

    @staticmethod
    def _load_cached_images(image_folder: str) -> List[Image.Image]:
        """
        Load previously rendered page images from the cache directory.

        Args:
            image_folder: Directory holding cached page PNGs.

        Returns:
            Cached images, or an empty list if the directory is absent or holds
            no PNGs. An empty result means "re-render": the directory existing is
            not on its own proof that a usable cache was written.
        """
        if not os.path.isdir(image_folder):
            return []

        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".png")],
            key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]),
        )
        return [Image.open(os.path.join(image_folder, f)) for f in image_files]

    def _get_prior_text(self, pdf_path: str) -> List[str]:
        """
        Extract text from each page of the PDF using selected backend.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of strings, one per page.
        """
        return self.digital_text_parser.extract_pages(pdf_path)

    def _pdf_image_to_base64_str(self, pdf_page: Image.Image) -> str:
        """
        Convert a PIL Image to base64 encoded JPEG string.

        Args:
            pdf_page: PIL Image object.

        Returns:
            Base64 encoded string.
        """
        image_buffer = io.BytesIO()
        pdf_page.save(image_buffer, format="JPEG")
        byte_data = image_buffer.getvalue()
        return base64.b64encode(byte_data).decode("utf-8")

    @retry(
        wait=_wait_for_exception,
        stop=_stop_for_exception,
        reraise=True,
    )
    def _process_image_with_provider(
        self,
        image_base64: str,
        prior_text: Optional[str] = None,
    ) -> "VisionResult":
        """
        Send image to LLM provider for processing.

        Args:
            image_base64: Base64-encoded image string.
            prior_text: Optional prior text for context.

        Returns:
            VisionResult with text and token usage.
        """
        vision_base = (
            "Write a Markdown version of this page keeping as much of the "
            "semantic meaning from information hierarchy as possible. For "
            "tabular-like data (including chart data), make easy to read tables "
            "as they'd be presented by a financial analyst.\n\n"
            "DO NOT include any 'meta description' of the markdown itself, like:"
            "\n- 'In the tables, the data should reflect the values provided in "
            "the original image.'"
            "\n- 'This markdown version maintains the hierarchy and clarity of the "
            "original page using headers and tables to present the financial data "
            "in an analyst-friendly format.'"
            "\n- 'In this Markdown version, the hierarchy of information is "
            "preserved with headers (`#`, `##`, `###`) and tables are created "
            "for easier readability as per the data presented.'\n"
            "Do NOT start each page with ```markdown or end with ```."
        )

        vision_assist = (
            "\n\nYour vision isn't great, so I've provided previously extracted "
            "text to help in <prior_text> tags. That text isn't perfect either so "
            "use a balanced approach to create the full Markdown output.\n"
            "\n<prior_text>\n{prior_text}\n</prior_text>\n"
        )

        prompt = f"{vision_base}{vision_assist}" if prior_text else vision_base

        return self.provider.process_vision(
            image_base64=image_base64,
            prompt=prompt,
            prior_text=prior_text,
            model=self.model_id,
            max_tokens=4096,
        )

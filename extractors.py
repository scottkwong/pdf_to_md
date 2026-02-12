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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from pdf2image import convert_from_path
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from digital_text_parsers import (
    BaseDigitalTextParser,
    create_digital_text_parser,
)

if TYPE_CHECKING:
    from llm_providers import BaseProvider


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
        max_parallel_pages: int = 10,
        digital_text_parser: str = "auto",
    ):
        """
        Initialize VisionExtractor.

        Args:
            provider: LLM provider instance (from llm_providers module).
            model_id: Model identifier for the provider.
            mode: Processing mode - 'v' for vision-only, 'vt' for vision-and-text.
            max_parallel_pages: Maximum number of pages to process in parallel.
            digital_text_parser: Parser engine used for first-pass digital text
                parsing in 'vt' mode. Valid values are auto, pypdf, pymupdf.
        """
        self.provider = provider
        self.model_id = model_id
        self.mode = mode
        self.max_parallel_pages = max_parallel_pages
        self._digital_text_parser_selection = create_digital_text_parser(
            digital_text_parser
        )
        self.digital_text_parser: BaseDigitalTextParser = (
            self._digital_text_parser_selection.parser
        )

        # Validate mode
        if mode not in ["v", "vt"]:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are ['v', 'vt'].")

    @property
    def name(self) -> str:
        return f"Vision ({self.model_id})"

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
        pdf_file_name = os.path.basename(pdf_path)

        # Load page images and prior text
        images, prior_texts = self._load_page_data(pdf_path, output_dir)

        # Process all pages in parallel
        results, failed_pages = self._process_pages_parallel(
            images, prior_texts, pdf_file_name, verbose
        )

        # Log any errors
        self._log_processing_errors(failed_pages, len(images), results)

        # Assemble final markdown in page order
        markdown_content = self._assemble_markdown(
            results, len(images), pdf_file_name
        )

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
    ) -> tuple[dict[int, str], list[tuple[int, str]]]:
        """
        Process all pages in parallel using ThreadPoolExecutor.

        Args:
            images: List of page images.
            prior_texts: List of prior text for each page.
            pdf_file_name: Name of the PDF file for headers.
            verbose: Whether to show progress bar.

        Returns:
            Tuple of (results dict, failed_pages list).
            results: Maps page index to markdown content.
            failed_pages: List of (page_index, error_message) tuples.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_pages = len(images)
        results: dict[int, str] = {}
        failed_pages: list[tuple[int, str]] = []

        logger.debug(
            f"Starting parallel extraction: {total_pages} pages, "
            f"max_workers={self.max_parallel_pages}"
        )

        with ThreadPoolExecutor(max_workers=self.max_parallel_pages) as executor:
            futures = {
                executor.submit(
                    self._process_single_page, ix, img, txt, pdf_file_name,
                    total_pages
                ): ix
                for ix, (img, txt) in enumerate(zip(images, prior_texts))
            }

            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(
                    iterator, total=len(futures), desc="Processing pages"
                )

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
                    failed_pages.append((page_index, str(e)))
                    logger.error(
                        f"FAILED: Page {page_index + 1} failed to process: {e}"
                    )

        return results, failed_pages

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
        markdown_text = self._process_image_with_provider(image_base64, prior_text)

        logger.debug(
            f"Completed page {page_index + 1}/{total_pages} "
            f"(0-indexed: {page_index})"
        )

        page_header = f"File: {pdf_file_name}; Page: {page_index + 1}\n"
        return page_index, page_header + markdown_text

    def _log_processing_errors(
        self,
        failed_pages: list[tuple[int, str]],
        total_pages: int,
        results: dict[int, str],
    ) -> None:
        """
        Log errors for failed or missing pages.

        Args:
            failed_pages: List of (page_index, error_message) tuples.
            total_pages: Expected total number of pages.
            results: Dict of successfully processed pages.
        """
        if failed_pages:
            failed_nums = [p + 1 for p, _ in failed_pages]
            logger.error(
                f"PAGE PROCESSING ERRORS: {len(failed_pages)} of {total_pages} "
                f"pages failed to process. Failed pages: {failed_nums}"
            )
            for page_idx, error_msg in failed_pages:
                logger.error(f"  Page {page_idx + 1}: {error_msg}")

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

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
            images = convert_from_path(pdf_path, thread_count=os.cpu_count())
            for i, image in enumerate(images):
                image.save(os.path.join(image_folder, f"{base_name}_image_{i}.png"))
        else:
            image_files = sorted(
                [f for f in os.listdir(image_folder) if f.endswith(".png")],
                key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]),
            )
            images = [
                Image.open(os.path.join(image_folder, f)) for f in image_files
            ]

        return images

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
        wait=wait_random_exponential(min=1.0 / 5000, max=5),
        stop=stop_after_attempt(3),
    )
    def _process_image_with_provider(
        self,
        image_base64: str,
        prior_text: Optional[str] = None,
    ) -> str:
        """
        Send image to LLM provider for processing.

        Args:
            image_base64: Base64-encoded image string.
            prior_text: Optional prior text for context.

        Returns:
            Markdown text from the model.
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

"""
PDF extraction strategy abstraction layer.

Provides a unified interface for different PDF-to-markdown extraction methods:
- Vision-based extraction using LLM providers (existing functionality)
- LlamaParse API extraction (new functionality)

This module can be extended to support additional extraction methods.
"""
import base64
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

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
    ):
        """
        Initialize VisionExtractor.

        Args:
            provider: LLM provider instance (from llm_providers module).
            model_id: Model identifier for the provider.
            mode: Processing mode - 'v' for vision-only, 'vt' for vision-and-text.
            max_parallel_pages: Maximum number of pages to process in parallel.
        """
        self.provider = provider
        self.model_id = model_id
        self.mode = mode
        self.max_parallel_pages = max_parallel_pages

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
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pdf_file_name = os.path.basename(pdf_path)

        # Get images
        images = self._pdf_to_images_with_storage(pdf_path, output_dir)

        # Get prior texts
        if self.mode == "v":
            prior_texts = [None] * len(images)
        else:  # mode == 'vt'
            prior_texts = self._get_prior_text(pdf_path)

        # Check that lengths match
        if len(prior_texts) != len(images):
            raise ValueError(
                f"The number of prior texts ({len(prior_texts)}) does not match "
                f"the number of images ({len(images)})."
            )

        def process_page(
            page_index: int, image: "Image.Image", prior_text: Optional[str]
        ) -> tuple[int, str]:
            """
            Process a single page and return (index, content).

            Args:
                page_index: Zero-based page index.
                image: PIL Image of the page.
                prior_text: Optional extracted text for context.

            Returns:
                Tuple of (page_index, markdown_content_with_header).
            """
            image_base64 = self._pdf_image_to_base64_str(image)
            markdown_text = self._process_image_with_provider(
                image_base64,
                prior_text,
            )
            page_header = f"File: {pdf_file_name}; Page: {page_index + 1}\n"
            return page_index, page_header + markdown_text

        # Process pages in parallel
        results: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_parallel_pages) as executor:
            futures = {
                executor.submit(process_page, ix, img, txt): ix
                for ix, (img, txt) in enumerate(zip(images, prior_texts))
            }

            # Use tqdm for progress if verbose
            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(
                    iterator, total=len(futures), desc="Processing pages"
                )

            for future in iterator:
                page_index, content = future.result()
                results[page_index] = content

        # Reconstruct in page order
        markdown_content = [results[i] for i in range(len(results))]

        return ExtractionResult(
            markdown="\n".join(markdown_content),
            page_count=len(images),
            metadata={
                "extractor": "vision",
                "model": self.model_id,
                "mode": self.mode,
                "max_parallel_pages": self.max_parallel_pages,
            },
        )

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
            images = convert_from_path(pdf_path)
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
        Extract text from each page of the PDF using PyPDF2.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of strings, one per page.
        """
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text_list = [page.extract_text() for page in reader.pages]
        return text_list

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

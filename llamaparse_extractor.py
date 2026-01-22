"""
LlamaParse extraction implementation.

Uses LlamaIndex's LlamaParse API for document-level PDF extraction.
LlamaParse is a dedicated PDF-to-markdown parsing service that processes
entire documents at once, optimized for structured documents like
financial reports, scientific papers, and complex layouts.
"""
import os
from typing import Optional

from extractors import BaseExtractor, ExtractionResult


class LlamaParseExtractor(BaseExtractor):
    """
    LlamaParse-based extraction using LlamaCloud API.

    Processes entire PDF at once, returning complete markdown.
    Different from vision-based extraction which processes page-by-page.
    """

    # Available tiers mapping
    TIERS = {
        "fast": "fast",
        "cost_effective": "cost_effective",
        "agentic": "agentic",
        "agentic_plus": "agentic_plus",
    }
    DEFAULT_TIER = "agentic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        tier: str = DEFAULT_TIER,
        language: str = "en",
        verbose: bool = False,
    ):
        """
        Initialize LlamaParse extractor.

        Args:
            api_key: LlamaCloud API key. If None, reads from LLAMA_CLOUD_API_KEY.
            tier: Processing tier (fast, cost_effective, agentic, agentic_plus).
                - fast: Speed priority, best for simple documents
                - cost_effective: Budget-friendly for standard documents
                - agentic: Balanced accuracy and speed (default)
                - agentic_plus: Maximum fidelity for complex layouts
            language: Document language code (default: "en").
            verbose: Enable verbose logging from LlamaParse.
        """
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not found in environment. "
                "Set it in your .env file or pass api_key parameter. "
                "Get your API key from https://cloud.llamaindex.ai"
            )

        if tier not in self.TIERS:
            raise ValueError(
                f"Invalid tier '{tier}'. Valid tiers: {list(self.TIERS.keys())}"
            )

        self.tier = tier
        self.language = language
        self._verbose = verbose
        self._parser = None  # Lazy initialization

    @property
    def name(self) -> str:
        return f"LlamaParse ({self.tier})"

    def _get_parser(self):
        """Lazy initialization of LlamaParse parser."""
        if self._parser is None:
            try:
                from llama_parse import LlamaParse
            except ImportError:
                raise ImportError(
                    "llama-parse package not installed. "
                    "Run: pip install llama-parse"
                )

            self._parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=self._verbose,
                language=self.language,
            )
        return self._parser

    def extract(
        self,
        pdf_path: str,
        output_dir: str,
        verbose: bool = True,
    ) -> ExtractionResult:
        """
        Extract markdown from PDF using LlamaParse.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for intermediate files (unused by LlamaParse).
            verbose: Print progress information.

        Returns:
            ExtractionResult with complete markdown content.
        """
        pdf_file_name = os.path.basename(pdf_path)

        if verbose:
            print(f"Extracting '{pdf_file_name}' with LlamaParse ({self.tier})...")

        try:
            parser = self._get_parser()
            documents = parser.load_data(pdf_path)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str:
                raise RuntimeError(
                    f"LlamaParse rate limit exceeded. Try again later or "
                    f"use --extractor vision. Error: {e}"
                )
            elif "authentication" in error_str or "401" in error_str:
                raise ValueError(
                    "LlamaParse authentication failed. "
                    "Verify your LLAMA_CLOUD_API_KEY is correct."
                )
            raise

        # Combine all document chunks into single markdown
        # Add page markers for consistency with vision extractor
        markdown_parts = []
        for i, doc in enumerate(documents):
            page_header = f"File: {pdf_file_name}; Page: {i + 1}\n"
            markdown_parts.append(f"{page_header}{doc.text}")

        markdown = "\n\n---\n\n".join(markdown_parts)

        if verbose:
            print(f"Extracted {len(documents)} pages using LlamaParse ({self.tier})")

        return ExtractionResult(
            markdown=markdown,
            page_count=len(documents),
            metadata={
                "extractor": "llamaparse",
                "tier": self.tier,
                "language": self.language,
            },
        )

    async def extract_async(
        self,
        pdf_path: str,
        output_dir: str,
        verbose: bool = True,
    ) -> ExtractionResult:
        """
        Async version of extract for parallel processing.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory for intermediate files (unused).
            verbose: Print progress information.

        Returns:
            ExtractionResult with complete markdown content.
        """
        pdf_file_name = os.path.basename(pdf_path)

        if verbose:
            print(
                f"Extracting '{pdf_file_name}' with LlamaParse ({self.tier}) [async]..."
            )

        parser = self._get_parser()
        documents = await parser.aload_data(pdf_path)

        markdown_parts = []
        for i, doc in enumerate(documents):
            page_header = f"File: {pdf_file_name}; Page: {i + 1}\n"
            markdown_parts.append(f"{page_header}{doc.text}")

        markdown = "\n\n---\n\n".join(markdown_parts)

        if verbose:
            print(f"Extracted {len(documents)} pages using LlamaParse ({self.tier})")

        return ExtractionResult(
            markdown=markdown,
            page_count=len(documents),
            metadata={
                "extractor": "llamaparse",
                "tier": self.tier,
                "language": self.language,
            },
        )

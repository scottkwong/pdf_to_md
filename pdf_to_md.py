#!/usr/bin/env python3
"""
Converts a PDF file or folder to Markdown using LLM vision models.

The script takes an input PDF file path as an argument and produces a Markdown
file in the same directory with the same name as the PDF file. Supports
multiple LLM providers via OpenRouter and direct APIs.
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from typing import Optional, Tuple, TYPE_CHECKING


def _get_version() -> str:
    """Return package version from metadata, or 0.0.0.dev if not installed."""
    try:
        from importlib.metadata import version
        return version("pdf-to-md")
    except Exception:  # pragma: no cover
        return "0.0.0.dev"

if TYPE_CHECKING:
    from llm_providers import BaseProvider
    from extractors import BaseExtractor

# Load environment variables
load_dotenv()

# Import llm_providers after load_dotenv (will be imported later if needed)
# This allows --list-models to work without all dependencies


def _initialize_provider(
    model: str, prefer_openrouter: bool = True
    ) -> Tuple["BaseProvider", str]:
    """
    Initialize provider for the specified model with fallback logic.

    Args:
        model: Model identifier from models.json.
        prefer_openrouter: If True, prefer OpenRouter when available.

    Returns:
        Tuple of (provider_instance, model_id).
    """
    from llm_providers import create_provider

    try:
        provider, model_id = create_provider(model, prefer_openrouter)
        return provider, model_id
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def create_extractor(
    extractor_type: str,
    model: str = "gpt-5.2",
    mode: str = "vt",
    digital_text_parser: str = "auto",
    prefer_openrouter: bool = True,
    llamaparse_tier: str = "agentic",
    language: str = "en",
    verbose: bool = False,
    max_parallel_pages: int = 10,
) -> "BaseExtractor":
    """
    Create appropriate extractor based on type.

    Args:
        extractor_type: Type of extractor ('vision' or 'llamaparse').
        model: Model identifier for vision extractor.
        mode: Processing mode for vision extractor ('v' or 'vt').
        digital_text_parser: Parser engine for first-pass digital text parsing
            in vision+text mode. One of: auto, pypdf, pymupdf.
        prefer_openrouter: Whether to prefer OpenRouter for vision extractor.
        llamaparse_tier: Processing tier for LlamaParse extractor.
        language: Document language for LlamaParse extractor.
        verbose: Enable verbose logging.
        max_parallel_pages: Max pages to process in parallel (VisionExtractor).

    Returns:
        Configured BaseExtractor instance.
    """
    if extractor_type == "llamaparse":
        from llamaparse_extractor import LlamaParseExtractor
        return LlamaParseExtractor(
            tier=llamaparse_tier,
            language=language,
            verbose=verbose,
        )
    else:
        # Default: vision-based extraction
        from extractors import VisionExtractor
        provider, model_id = _initialize_provider(model, prefer_openrouter)
        return VisionExtractor(
            provider=provider,
            model_id=model_id,
            mode=mode,
            max_parallel_pages=max_parallel_pages,
            digital_text_parser=digital_text_parser,
        )


def pdf_to_markdown_with_extractor(
    pdf_path: str,
    output_dir: str,
    extractor: "BaseExtractor",
    verbose: bool = True,
) -> str:
    """
    Convert PDF to Markdown using the specified extractor.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output file.
        extractor: Configured extractor instance.
        verbose: Print progress to console.

    Returns:
        Path to the output markdown file.
    """
    output_file_name = os.path.basename(pdf_path).rsplit('.', 1)[0] + '.md'
    output_file_path = os.path.join(output_dir, output_file_name)

    # Extract using the configured extractor
    result = extractor.extract(pdf_path, output_dir, verbose)

    # Write results
    with open(output_file_path, 'w') as file:
        file.write(result.markdown)

    return output_file_path


def list_available_models() -> None:
    """
    List all available models from models.json and display them.
    
    Exits the program after displaying the models.
    """
    import json
    
    # Load models.json directly
    models_file = os.path.join(os.path.dirname(__file__), "models.json")
    try:
        with open(models_file, "r") as f:
            models_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: models.json not found at {models_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in models.json: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print("Available Models:")
    print("=" * 70)
    print()
    
    # Find default model
    default_model = None
    for model_name, model_config in models_config.items():
        if model_config.get("default", False):
            default_model = model_name
            break
    
    # Display all models
    for model_name, model_config in sorted(models_config.items()):
        provider = model_config.get("provider", "unknown")
        supports_vision = model_config.get("supports_vision", False)
        is_default = model_config.get("default", False)
        
        # Build display line
        display_name = model_name
        if is_default:
            display_name += " (DEFAULT)"
        
        vision_status = "✓ Vision" if supports_vision else "✗ No Vision"
        
        print(f"  {display_name}")
        print(f"    Provider:        {provider.title()}")
        print(f"    OpenRouter ID:   {model_config.get('openrouter_id', 'N/A')}")
        print(f"    Direct API ID:   {model_config.get('direct_id', 'N/A')}")
        print(f"    Vision Support:  {vision_status}")
        print()
    
    if default_model:
        print(f"Default model: {default_model}")
    else:
        print("No default model specified in models.json")
    print("=" * 70)


def print_cli_configuration(args: argparse.Namespace, model: str) -> None:
    """
    Print CLI configuration to the console.

    Args:
        args: Parsed command-line arguments.
        model: Model identifier being used.
    """
    print("=" * 70)
    print("CLI Configuration:")
    print("=" * 70)
    print(f"  Version:            {_get_version()}")
    print(f"  Target path:        {args.target_path}")
    print(f"  Output directory:   {args.output_dir or '(default: same as input)'}")
    print(f"  Extractor:          {args.extractor}")
    if args.extractor == "llamaparse":
        print(f"  LlamaParse tier:    {args.llamaparse_tier}")
        print(f"  Language:           {args.language}")
    else:
        print(f"  Mode:               {args.mode}")
        print(f"  Digital parser:     {args.digital_text_parser}")
        print(f"  Model:              {model}")
        print(f"  Provider override:  {args.provider or '(none - auto-detect)'}")
        print(f"  Prefer OpenRouter:  {not args.prefer_direct}")
        print(f"  Parallel pages:     {args.parallel}")
    if args.benchmark_digital_text_parsers:
        benchmark_target = args.benchmark_pdf or "(default generated fixtures)"
        print(f"  Benchmark mode:     {args.benchmark_digital_text_parsers}")
        print(f"  Benchmark runs:     {args.benchmark_runs}")
        print(f"  Benchmark target:   {benchmark_target}")
    print(f"  Verbose:            {args.verbose}")
    print(f"  Debug:              {args.debug}")
    print(f"  Recursive:          {args.recursive}")
    if args.recursive:
        file_mode = "sequential" if args.single else "parallel"
        print(f"  File processing:    {file_mode}")
    print()


def print_model_resolution(
    model: str, prefer_openrouter: bool
) -> None:
    """
    Print model resolution information based on API key availability.
    
    Args:
        model: Model identifier to resolve.
        prefer_openrouter: Whether to prefer OpenRouter when available.
    """
    from llm_providers import (
        OpenRouterProvider,
        create_provider,
        get_available_providers,
        load_models_config,
    )
    
    print("=" * 70)
    print("Model Resolution:")
    print("=" * 70)
    available_providers = get_available_providers()
    print("Available API keys:")
    for provider, available in available_providers.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {provider:15} {status}")
    print()

    # Resolve model
    try:
        provider, model_id = create_provider(model, prefer_openrouter)
        models_config = load_models_config()
        model_info = models_config.get(model, {})
        provider_name = model_info.get("provider", "unknown")
        
        # Check if using OpenRouter
        using_openrouter = isinstance(provider, OpenRouterProvider)
        
        print(f"Model resolution:")
        print(f"  Requested:         {model}")
        print(f"  Actual model ID:   {model_id}")
        print(f"  Provider:          {provider_name}")
        print(f"  Via OpenRouter:   {using_openrouter}")
    except Exception as e:
        print(f"  ✗ Error resolving model: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print()


def print_digital_text_parser_resolution(requested_parser: str) -> None:
    """Print requested and resolved digital text parser details.

    Args:
        requested_parser: Parser requested from CLI configuration.
    """
    from digital_text_parsers import (
        create_digital_text_parser,
        get_available_digital_text_parsers,
    )

    try:
        selection = create_digital_text_parser(requested_parser)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    available_parsers = ", ".join(get_available_digital_text_parsers())
    print("=" * 70)
    print("Digital Text Parser Resolution:")
    print("=" * 70)
    print(f"  Requested parser: {selection.requested_parser}")
    print(f"  Resolved parser:  {selection.resolved_parser}")
    print(f"  Available:        {available_parsers}")
    print("=" * 70)
    print()


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    extractor: "BaseExtractor",
    verbose: bool,
) -> None:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Output directory for the markdown file.
        extractor: Configured extractor to use.
        verbose: Whether to print markdown to console.
    """
    output_file = pdf_to_markdown_with_extractor(
        pdf_path,
        output_dir,
        extractor,
        verbose,
    )
    print(f"Output file: {output_file}")


def process_directory_sequential(
    target_path: str,
    output_dir: Optional[str],
    extractor: "BaseExtractor",
    verbose: bool,
) -> None:
    """
    Process all PDF files in a directory sequentially.

    Args:
        target_path: Path to the directory.
        output_dir: Base output directory (None to use same as each PDF).
        extractor: Configured extractor to use.
        verbose: Whether to print markdown to console.
    """
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_output_dir = output_dir or os.path.dirname(pdf_path)
                process_single_pdf(
                    pdf_path,
                    pdf_output_dir,
                    extractor,
                    verbose,
                )


def process_directory_parallel(
    target_path: str,
    output_dir: Optional[str],
    extractor: "BaseExtractor",
    verbose: bool,
) -> None:
    """
    Process all PDF files in a directory in parallel.

    Args:
        target_path: Path to the directory.
        output_dir: Base output directory (None to use same as each PDF).
        extractor: Configured extractor to use.
        verbose: Whether to print markdown to console.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdf_output_dir = output_dir or os.path.dirname(pdf_path)
                    futures.append(
                        executor.submit(
                            process_single_pdf,
                            pdf_path,
                            pdf_output_dir,
                            extractor,
                            verbose,
                        )
                    )
        for future in futures:
            future.result()


def validate_file_path(file_path: str) -> None:
    """
    Validate that the file path exists and is a file.
    
    Args:
        file_path: Path to validate.
        
    Exits:
        sys.exit(1) if path is invalid.
    """
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)


def validate_directory_path(dir_path: str) -> None:
    """
    Validate that the directory path exists and is a directory.
    
    Args:
        dir_path: Path to validate.
        
    Exits:
        sys.exit(1) if path is invalid.
    """
    if not os.path.isdir(dir_path):
        print(f"Error: The path '{dir_path}' is not a directory.")
        sys.exit(1)


def main() -> None:
    """CLI entry point: parse arguments and run the PDF-to-Markdown pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to a Markdown file using LLM vision models."
    )
    parser.add_argument(
        "target_path",
        type=str,
        nargs="?",
        help="The path to the input PDF file or directory containing PDF "
        "files.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and show the default, then exit.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="The directory to the output files. If not specified, defaults "
        "to the same location as the PDF file.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="vt",
        choices=["v", "vt"],
        help="Toggle between 'v' for vision-only and 'vt' (default) for "
        "vision-and-text processing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model identifier from models.json to use for both vision and text "
        "processing (default: gpt-5.2).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "openai", "anthropic", "google"],
        default=None,
        help="Force specific provider (optional).",
    )
    parser.add_argument(
        "--prefer-direct",
        action="store_true",
        default=False,
        help="Skip OpenRouter and use direct APIs only.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        default=True,
        dest="verbose",
        help="If set, do not print the markdown text to the screen.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="If set, treat the target path as a directory and process all "
        "PDF files within it recursively.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        nargs="?",
        const=10,
        default=10,
        help="Max parallel pages to process with VisionExtractor (default: 10).",
    )
    parser.add_argument(
        "-s",
        "--single",
        action="store_true",
        default=False,
        help="Process files sequentially instead of in parallel when using -r.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["vision", "llamaparse"],
        default="vision",
        help="Extraction method: 'vision' (LLM-based, default) or 'llamaparse' "
        "(LlamaCloud API).",
    )
    parser.add_argument(
        "--llamaparse-tier",
        type=str,
        choices=["fast", "cost_effective", "agentic", "agentic_plus"],
        default="agentic",
        help="LlamaParse processing tier (only used with --extractor llamaparse). "
        "Options: fast, cost_effective, agentic (default), agentic_plus.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Document language code for LlamaParse (default: 'en').",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging for page processing order.",
    )
    parser.add_argument(
        "--digital-text-parser",
        type=str,
        choices=["auto", "pypdf", "pymupdf"],
        default="auto",
        help="Parser engine for first-pass digital text parsing in mode 'vt'. "
        "Defaults to 'auto' (prefers PyMuPDF when installed).",
    )
    parser.add_argument(
        "--benchmark-digital-text-parsers",
        action="store_true",
        default=False,
        help="Run a non-LLM benchmark that compares digital text parser "
        "engines (pypdf vs PyMuPDF) and exit.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of extraction runs per backend when benchmark mode is "
        "enabled (default: 10).",
    )
    parser.add_argument(
        "--benchmark-pdf",
        type=str,
        default=None,
        help="Optional PDF path override for benchmark mode. If omitted, "
        "generated fixture PDFs are used.",
    )
    args = parser.parse_args()

    # Configure logging - always show errors, debug only with -d flag
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet pypdf "Ignoring wrong pointing object" (benign PDF quirks)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    # Set extractors logger to show errors by default, debug with -d
    extractors_logger = logging.getLogger("extractors")
    if args.debug:
        extractors_logger.setLevel(logging.DEBUG)
    else:
        extractors_logger.setLevel(logging.ERROR)
    
    # Handle --list-models and --version (before heavy imports)
    if args.list_models:
        list_available_models()
        sys.exit(0)
    if args.version:
        print(f"pdf_to_md {_get_version()}")
        sys.exit(0)

    # Require target_path if not listing models
    if not args.target_path and not args.benchmark_digital_text_parsers:
        parser.error("target_path is required unless using --list-models")

    # Extract configuration from args
    model = args.model
    prefer_openrouter = not args.prefer_direct

    # Print configuration
    print_cli_configuration(args, model)

    if args.extractor == "vision":
        print_digital_text_parser_resolution(args.digital_text_parser)

    if args.benchmark_digital_text_parsers:
        from benchmark_digital_text_parsers import (
            benchmark_digital_text_parsers,
            print_benchmark_report,
        )
        summary = benchmark_digital_text_parsers(
            runs=args.benchmark_runs,
            pdf_path=args.benchmark_pdf,
        )
        print_benchmark_report(summary)
        sys.exit(0)

    # Create extractor and handle model resolution
    extractor = None
    if args.extractor == "llamaparse":
        try:
            extractor = create_extractor(
                extractor_type="llamaparse",
                llamaparse_tier=args.llamaparse_tier,
                language=args.language,
                verbose=args.verbose,
            )
            print("=" * 70)
            print("LlamaParse Configuration:")
            print("=" * 70)
            print(f"  Extractor:          {extractor.name}")
            print(f"  Tier:               {args.llamaparse_tier}")
            print(f"  Language:           {args.language}")
            print("=" * 70)
            print()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Vision-based extraction - resolve model
        print_model_resolution(model, prefer_openrouter)
        extractor = create_extractor(
            extractor_type="vision",
            model=model,
            mode=args.mode,
            digital_text_parser=args.digital_text_parser,
            prefer_openrouter=prefer_openrouter,
            verbose=args.verbose,
            max_parallel_pages=args.parallel,
        )

    # Process PDF(s) based on mode
    if args.recursive:
        validate_directory_path(args.target_path)
        if args.single:
            # Sequential processing when -s/--single is specified
            process_directory_sequential(
                args.target_path,
                args.output_dir,
                extractor,
                args.verbose,
            )
        else:
            # Default: parallel file processing
            process_directory_parallel(
                args.target_path,
                args.output_dir,
                extractor,
                args.verbose,
            )
    else:
        validate_file_path(args.target_path)
        output_dir = args.output_dir or os.path.dirname(args.target_path)
        process_single_pdf(
            args.target_path,
            output_dir,
            extractor,
            args.verbose,
        )


if __name__ == "__main__":
    main()

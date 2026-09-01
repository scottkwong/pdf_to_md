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

from models_config import ModelKey, Provider


def _get_version() -> str:
    """Return package version from metadata, or 0.0.0.dev if not installed."""
    try:
        from importlib.metadata import version
        return version("pdf-to-md")
    except Exception:  # pragma: no cover
        return "0.0.0.dev"


# Overall default model when neither --model nor PDF_TO_MD_MODEL is set. To make
# a different model your personal default, set PDF_TO_MD_MODEL in your .env
# (e.g. PDF_TO_MD_MODEL=qwen3.7-plus) along with the matching provider key.
DEFAULT_MODEL = ModelKey.GPT_5_5.value
_FIREWORKS_FALLBACK_MODEL = ModelKey.QWEN_3_7_PLUS.value


def _default_model() -> str:
    """Return the default model, honoring the PDF_TO_MD_MODEL override."""
    return os.getenv("PDF_TO_MD_MODEL", DEFAULT_MODEL)


def _default_fireworks_model() -> str:
    """Return the default Fireworks model (the one flagged in models.json)."""
    try:
        from models_config import load_models_config
        for name, cfg in load_models_config().items():
            if cfg.get("fireworks_default"):
                return name
    except Exception:  # pragma: no cover - fall back if config unavailable
        pass
    return _FIREWORKS_FALLBACK_MODEL

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
    model: str = "gpt-5.5",
    mode: str = "vt",
    digital_text_parser: str = "auto",
    prefer_openrouter: bool = False,
    llamaparse_tier: str = "agentic",
    language: str = "en",
    verbose: bool = False,
    max_parallel_pages: Optional[int] = None,
    local_model: str = "qwen2.5vl:7b",
    ollama_url: str = "http://localhost:11434/api/generate",
    local_num_ctx: int = 16384,
    local_num_predict: int = 8192,
) -> "BaseExtractor":
    """
    Create appropriate extractor based on type.

    Args:
        extractor_type: Type of extractor ('vision', 'llamaparse', or 'local').
        model: Model identifier for vision extractor.
        mode: Processing mode for vision extractor ('v' or 'vt').
        digital_text_parser: Parser engine for first-pass digital text parsing
            in vision+text mode. One of: auto, pypdf, pymupdf.
        prefer_openrouter: Whether to prefer OpenRouter for vision extractor.
        llamaparse_tier: Processing tier for LlamaParse extractor.
        language: Document language for LlamaParse extractor.
        verbose: Enable verbose logging.
        max_parallel_pages: Max pages to process in parallel (VisionExtractor).
            None resolves to the provider's safe concurrency.
        local_model: Ollama model tag for the local extractor.
        ollama_url: Ollama generate endpoint for the local extractor.
        local_num_ctx: Context window in tokens for the local extractor.
        local_num_predict: Output token budget per page for the local extractor.

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
    elif extractor_type == "local":
        # Offline extraction against a locally running Ollama server.
        from extractors import VisionExtractor
        from local_ocr import LocalOllamaProvider, ensure_ollama_model

        # Pull the model up front so weights download with visible progress
        # before any page processing begins.
        ensure_ollama_model(local_model, ollama_url, verbose=True)
        provider = LocalOllamaProvider(
            model_name=local_model,
            base_url=ollama_url,
            num_ctx=local_num_ctx,
            num_predict=local_num_predict,
        )
        return VisionExtractor(
            provider=provider,
            model_id=local_model,
            mode=mode,
            max_parallel_pages=max_parallel_pages,
            digital_text_parser=digital_text_parser,
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

    # Print LLM cost summary if available
    meta = result.metadata or {}
    total_input = meta.get("total_input_tokens", 0)
    total_output = meta.get("total_output_tokens", 0)
    total_cost = meta.get("total_cost_usd", 0.0)

    if total_input or total_output:
        print(f"\nLLM Cost Summary:")
        print(f"  Model:         {meta.get('model', 'unknown')}")
        print(f"  Input tokens:  {total_input:,}")
        print(f"  Output tokens: {total_output:,}")
        if total_cost > 0:
            print(f"  Total cost:    ${total_cost:.4f}")
        else:
            print(f"  Total cost:    (pricing unavailable)")

    return output_file_path


def list_available_models() -> None:
    """
    List all available models from models.json and display them.

    Exits the program after displaying the models.
    """
    import json
    try:
        from models_config import load_models_config
        models_config = load_models_config()
    except FileNotFoundError as e:
        print(f"Error: models.json not found: {e}")
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
    effective_extractor = "local" if args.local else args.extractor
    print(f"  Extractor:          {effective_extractor}")
    if args.local:
        print(f"  Local model:        {args.local_model}")
        print(f"  Ollama URL:         {args.ollama_url}")
        print(f"  Context window:     {args.local_num_ctx} tokens")
        print(f"  Output budget:      {args.local_num_predict} tokens/page")
        print(f"  Mode:               {args.mode}")
        print(f"  Digital parser:     {args.digital_text_parser}")
        print(
            f"  Parallel pages:     "
            f"{args.parallel if args.parallel else 'auto (provider default)'}"
        )
    elif args.extractor == "llamaparse":
        print(f"  LlamaParse tier:    {args.llamaparse_tier}")
        print(f"  Language:           {args.language}")
    else:
        print(f"  Mode:               {args.mode}")
        print(f"  Digital parser:     {args.digital_text_parser}")
        print(f"  Model:              {model}")
        print(f"  Provider override:  {args.provider or '(none - auto-detect)'}")
        print(f"  Prefer OpenRouter:  {args.prefer_openrouter}")
        print(
            f"  Parallel pages:     "
            f"{args.parallel if args.parallel else 'auto (provider default)'}"
        )
    if args.benchmark:
        benchmark_target = (
            args.benchmark_pdf
            or (args.target_path if args.target_path else "(default generated fixtures)")
        )
        models_desc = args.benchmark_models or "(default: top OpenAI, Anthropic, Local)"
        print(f"  Benchmark mode:     model comparison")
        print(f"  Benchmark models:   {models_desc}")
        print(f"  Benchmark target:   {benchmark_target}")
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
    model: str, prefer_openrouter: bool, max_parallel_pages: Optional[int] = None
) -> None:
    """
    Print model resolution information based on API key availability.
    
    Args:
        model: Model identifier to resolve.
        prefer_openrouter: Whether to prefer OpenRouter when available.
        max_parallel_pages: Explicit --parallel value, or None to report the
            concurrency the resolved provider declares as safe.
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

        from extractors import _provider_safe_concurrency

        if max_parallel_pages is None:
            concurrency = _provider_safe_concurrency(provider)
            origin = f"{provider_name} default"
        else:
            concurrency = max_parallel_pages
            origin = "explicit --parallel"
        print(f"  Page concurrency:  {concurrency} ({origin})")
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
        default=_default_model(),
        help="Model identifier from models.json to use for both vision and text "
        "processing. Defaults to gpt-5.5, or the PDF_TO_MD_MODEL env var if set.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=[provider.value for provider in Provider],
        default=None,
        help="Force specific provider (optional).",
    )
    parser.add_argument(
        "--prefer-openrouter",
        action="store_true",
        default=False,
        help="Route via OpenRouter when available. Default is to use direct "
        "provider APIs.",
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
        default=None,
        help=(
            "Max parallel pages to process with VisionExtractor. Defaults to "
            "the resolved provider's safe concurrency (10 for most, 3 for "
            "Fireworks, whose per-minute quota 429s a wider fan-out)."
        ),
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
        "(LlamaCloud API). Ignored when --local is set.",
    )
    parser.add_argument(
        "--fireworks",
        action="store_true",
        default=False,
        help="Use Fireworks AI as the provider with a capable multimodal model "
        "(default: qwen3.7-plus). Requires FIREWORKS_API_KEY. Shortcut for "
        "'--model <fireworks-model>'; pick the model with --fireworks-model.",
    )
    parser.add_argument(
        "--fireworks-model",
        type=str,
        default=_default_fireworks_model(),
        help="Fireworks model (models.json key) to use with --fireworks "
        "(default: qwen3.7-plus).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run OCR fully offline against a local Ollama server. No API "
        "keys or per-token cost.",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="qwen2.5vl:7b",
        help="Ollama vision model tag to use with --local and as the local "
        "entry in --benchmark (default: qwen2.5vl:7b).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint for --local / local benchmark models "
        "(default: http://localhost:11434/api/generate).",
    )
    parser.add_argument(
        "--local-num-ctx",
        type=int,
        default=16384,
        help="Context window in tokens for --local / local benchmark models "
        "(default: 16384). Sized for one page plus its prior text; raising it "
        "increases memory use sharply, since the KV cache scales with it.",
    )
    parser.add_argument(
        "--local-num-predict",
        type=int,
        default=8192,
        help="Output token budget per page for --local / local benchmark "
        "models (default: 8192). Reasoning models such as qwen3-vl spend this "
        "budget thinking before they answer, and return an empty page if it "
        "runs out; non-reasoning models stop early and are unaffected.",
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
        "--benchmark",
        action="store_true",
        default=False,
        help="Benchmark and compare the top model from each provider "
        "(OpenAI, Anthropic, and Local) on the same PDF, then exit. Runs "
        "models in parallel and streams progress. The local model may need "
        "to download weights on first run.",
    )
    parser.add_argument(
        "--benchmark-models",
        type=str,
        default=None,
        help="Comma-separated override for --benchmark, e.g. "
        "'gpt-5.5,claude-opus-4.6,local'. Use 'local' or 'local:<model>' for "
        "an Ollama model; other entries are models.json keys. Defaults to the "
        "top OpenAI, Anthropic, and Local models.",
    )
    parser.add_argument(
        "--benchmark-reference",
        type=str,
        default=None,
        help="Model label to use as the baseline in the --benchmark HTML "
        "comparison; every other model is diffed against it and it is shown "
        "first. Defaults to the first benchmarked model. Remaining models are "
        "ordered cheapest-first.",
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

    # Require target_path unless in a mode that supplies its own input
    if (
        not args.target_path
        and not args.benchmark_digital_text_parsers
        and not args.benchmark
    ):
        parser.error("target_path is required unless using --list-models")

    # --fireworks and --local both select where inference runs; refuse both.
    if args.local and args.fireworks:
        parser.error("--local and --fireworks are mutually exclusive.")

    # Extract configuration from args. --fireworks is a shortcut that swaps in a
    # Fireworks vision model, which then resolves to FireworksProvider normally.
    model = args.fireworks_model if args.fireworks else args.model
    prefer_openrouter = args.prefer_openrouter

    # Print configuration
    print_cli_configuration(args, model)

    # Fail fast with a clear message rather than dropping into the interactive
    # fallback menu when --fireworks is used without a key.
    if args.fireworks:
        from llm_providers import get_available_providers
        if not get_available_providers().get(Provider.FIREWORKS):
            print(
                "Error: --fireworks requires FIREWORKS_API_KEY. Get a key at "
                "https://fireworks.ai and set it in your .env or environment."
            )
            sys.exit(1)

    if args.benchmark:
        from benchmark_models import (
            _resolve_benchmark_pdf,
            benchmark_models,
            default_benchmark_specs,
            parse_benchmark_specs,
            print_model_benchmark_report,
        )
        from benchmark_report import write_benchmark_html

        if args.benchmark_models:
            specs = parse_benchmark_specs(
                args.benchmark_models,
                local_model=args.local_model,
                ollama_url=args.ollama_url,
                prefer_openrouter=prefer_openrouter,
                num_ctx=args.local_num_ctx,
                num_predict=args.local_num_predict,
            )
        else:
            specs = default_benchmark_specs(
                local_model=args.local_model,
                ollama_url=args.ollama_url,
                prefer_openrouter=prefer_openrouter,
                num_ctx=args.local_num_ctx,
                num_predict=args.local_num_predict,
            )

        # Prefer an explicit benchmark PDF, then a target file, else fixtures.
        benchmark_pdf = args.benchmark_pdf
        if not benchmark_pdf and args.target_path and os.path.isfile(
            args.target_path
        ):
            benchmark_pdf = args.target_path
        benchmark_pdf = _resolve_benchmark_pdf(benchmark_pdf)

        # A benchmark always lands its artifacts, so the markdown and the HTML
        # comparison are there to inspect without needing -o.
        benchmark_dir = args.output_dir or os.path.join(
            os.getcwd(), "benchmark_output"
        )

        results = benchmark_models(
            specs=specs,
            pdf_path=benchmark_pdf,
            output_dir=benchmark_dir,
            max_parallel_pages=args.parallel,
        )
        print_model_benchmark_report(results)

        stem = os.path.basename(benchmark_pdf).rsplit(".", 1)[0]
        html_path = write_benchmark_html(
            results=results,
            pdf_path=benchmark_pdf,
            output_path=os.path.join(benchmark_dir, f"{stem}.comparison.html"),
            reference_label=args.benchmark_reference,
        )
        if html_path:
            # A file:// URL so the terminal renders it as a clickable link;
            # the plain path is kept for copy/paste and for piping to `open`.
            from pathlib import Path

            url = Path(html_path).resolve().as_uri()
            print()
            print("=" * 88)
            print("Visual comparison (PDF page vs each model, side by side):")
            print(f"  {url}")
            print(f"  open {html_path}")
            print("=" * 88)
        sys.exit(0)

    if args.extractor == "vision" and not args.local:
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
    if args.local:
        print_digital_text_parser_resolution(args.digital_text_parser)
        try:
            extractor = create_extractor(
                extractor_type="local",
                mode=args.mode,
                digital_text_parser=args.digital_text_parser,
                language=args.language,
                verbose=args.verbose,
                max_parallel_pages=args.parallel,
                local_model=args.local_model,
                ollama_url=args.ollama_url,
                local_num_ctx=args.local_num_ctx,
                local_num_predict=args.local_num_predict,
            )
            print("=" * 70)
            print("Local OCR Configuration (Ollama):")
            print("=" * 70)
            print(f"  Extractor:          {extractor.name}")
            print(f"  Local model:        {args.local_model}")
            print(f"  Ollama URL:         {args.ollama_url}")
            print(f"  Mode:               {args.mode}")
            print("=" * 70)
            print()
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.extractor == "llamaparse":
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
        print_model_resolution(model, prefer_openrouter, args.parallel)
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

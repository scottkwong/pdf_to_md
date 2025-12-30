#!/opt/homebrew/anaconda3/envs/pdf_to_md/bin/python
"""
Converts a PDF file or folder to Markdown using LLM vision models.

The script takes an input PDF file path as an argument and produces a Markdown
file in the same directory with the same name as the PDF file. Supports
multiple LLM providers via OpenRouter and direct APIs.
"""
import argparse
import base64
import io
import os
import sys

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_providers import BaseProvider

# Load environment variables
load_dotenv()

# Import llm_providers after load_dotenv (will be imported later if needed)
# This allows --list-models to work without all dependencies


def pdf_to_markdown(
        pdf_path: str,
        output_dir: str,
        mode: str = "vt",
        verbose: bool = True,
        vision_model: str = "gpt-5.2",
        text_model: Optional[str] = None,
        prefer_openrouter: bool = True,
    ) -> str:
    """
    Main function to convert a PDF to Markdown using LLM vision models.

    This function takes a path to a PDF file, an output directory, a mode
    indicating whether to use 'vision-only' (v) or 'vision-and-text' (vt)
    processing, and a verbose flag. It converts the PDF to images, processes
    each image with the specified vision model to generate markdown text, and
    writes the markdown text to a file with the same name as the PDF file but
    with a .md extension, located in the output directory. If verbose is True,
    it also prints the markdown text to the screen.

    Args:
        pdf_path: The path to the input PDF file.
        output_dir: The directory where the output markdown file will be saved.
        mode: The processing mode ('v' for vision-only, 'vt' for
            vision-and-text).
        verbose: If True, print the markdown text to the screen.
        vision_model: Model identifier from models.json for vision processing.
        text_model: Model identifier for text processing (defaults to
            vision_model).
        prefer_openrouter: If True, prefer OpenRouter when available.

    Returns:
        output_file_path
    """
    # Setup constants and validation
    output_file_name = os.path.basename(pdf_path).rsplit('.', 1)[0] + '.md'
    output_file_path = os.path.join(output_dir, output_file_name)
    pdf_file_name = os.path.basename(pdf_path)

    # Validate mode
    valid_modes = ['v', 'vt']
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are {valid_modes}."
        )

    # Initialize vision provider
    vision_provider, vision_model_id = _initialize_provider(
        vision_model, prefer_openrouter
    )

    # Get images
    images = _pdf_to_images_with_storage(pdf_path, output_dir)

    # Get prior texts
    if mode == 'v':
        prior_texts = [None] * len(images)
    elif mode == 'vt':
        prior_texts = _get_prior_text(pdf_path)

    # Check that lengths match
    if len(prior_texts) != len(images):
        raise ValueError(
            f"The number of prior texts ({len(prior_texts)}) does not match "
            f"the number of images ({len(images)})."
        )

    # Build the markdown
    markdown_content = []
    for ix, (image, prior_text) in enumerate(tqdm(zip(images, prior_texts))):
        image_base64 = _pdf_image_to_base64_str(image)
        markdown_text = _process_image_with_provider(
            vision_provider,
            vision_model_id,
            image_base64,
            prior_text,
        )
        markdown_text = (
            f"File: {pdf_file_name}; Page: {ix + 1}\n"
        ) + markdown_text
        markdown_content.append(markdown_text)
        if verbose:
            print(markdown_text)

    # Write results
    with open(output_file_path, 'w') as file:
        file.write('\n'.join(markdown_content))

    return output_file_path


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


@retry(
    wait=wait_random_exponential(min=1.0 / 5000, max=5),
    stop=stop_after_attempt(3),
)
def _process_image_with_provider(
    provider: "BaseProvider",
    model_id: str,
    image_base64: str,
    prior_text: Optional[str] = None,
) -> str:
    """
    Send a base64-encoded image to LLM provider for processing.

    Constructs a prompt for the model to interpret the image as a Markdown
    document, preserving the semantic meaning and information hierarchy,
    including tables. If prior text is provided, it is included to assist the
    model in the interpretation.

    Args:
        provider: Provider instance to use for processing.
        model_id: Model identifier for the provider.
        image_base64: The base64-encoded image to be processed.
        prior_text: Optional; previously extracted text to provide context.

    Returns:
        The Markdown version of the image content as interpreted by the model.
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

    return provider.process_vision(
        image_base64=image_base64,
        prompt=prompt,
        prior_text=prior_text,
        model=model_id,
        max_tokens=4096,
    )


def _pdf_to_images_with_storage(
        pdf_path: str, 
        output_dir: str
    ) -> List[Image.Image]:
    """
    Load images from the output directory if they exist, otherwise convert the 
    PDF to images and save them to the specified output directory.

    Args:
        pdf_path: The path to the input PDF file.
        output_dir: The directory where the output images will be saved.

    Returns:
        A list of PIL Image objects.
    """
    base_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
    image_folder = os.path.join(output_dir, base_name + '_images')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image.save(os.path.join(image_folder, f'{base_name}_image_{i}.png'))
    else:
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith('.png')],
            key=lambda x: int(x.rsplit('_', 1)[-1].split('.')[0])
        )
        images = [
            Image.open(os.path.join(image_folder, f)) for f in image_files
        ]
    return images


def _get_prior_text(pdf_path: str) -> List[str]:
    """
    Extracts simple text from each page of the PDF using PyPDF2.

    Args:
        pdf_path (str): The path to the input PDF file.

    Returns:
        List[str]: A list of strings where each string represents the extracted 
            text from a single page of the PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text_list = [page.extract_text() for page in reader.pages]
    return text_list


def _pdf_image_to_base64_str(pdf_page: Image) -> str:
    """
    Convert a PDF page to a base64 encoded JPEG image.

    Args:
        pdf_page (Image): A PIL Image object representing a PDF page.

    Returns:
        str: A base64 encoded string of the JPEG image.
    """
    image_buffer = io.BytesIO()
    pdf_page.save(image_buffer, format='JPEG')
    byte_data = image_buffer.getvalue()
    return base64.b64encode(byte_data).decode('utf-8')


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
    print(f"  Target path:        {args.target_path}")
    print(f"  Output directory:   {args.output_dir or '(default: same as input)'}")
    print(f"  Mode:               {args.mode}")
    print(f"  Verbose:            {args.verbose}")
    print(f"  Recursive:          {args.recursive}")
    print(f"  Parallel:           {args.parallel}")
    print(f"  Model:              {model}")
    print(f"  Provider override:  {args.provider or '(none - auto-detect)'}")
    print(f"  Prefer OpenRouter:  {not args.prefer_direct}")
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


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    processing_mode: str,
    verbose: bool,
    model: str,
    prefer_openrouter: bool,
) -> None:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Output directory for the markdown file.
        processing_mode: Processing mode ('v' or 'vt').
        verbose: Whether to print markdown to console.
        model: Model identifier to use.
        prefer_openrouter: Whether to prefer OpenRouter.
    """
    output_file = pdf_to_markdown(
        pdf_path,
        output_dir,
        processing_mode,
        verbose,
        model,
        None,  # text_model - same as vision_model
        prefer_openrouter,
    )
    print(f"Output file: {output_file}")


def process_directory_sequential(
    target_path: str,
    output_dir: Optional[str],
    processing_mode: str,
    verbose: bool,
    model: str,
    prefer_openrouter: bool,
) -> None:
    """
    Process all PDF files in a directory sequentially.
    
    Args:
        target_path: Path to the directory.
        output_dir: Base output directory (None to use same as each PDF).
        processing_mode: Processing mode ('v' or 'vt').
        verbose: Whether to print markdown to console.
        model: Model identifier to use.
        prefer_openrouter: Whether to prefer OpenRouter.
    """
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_output_dir = output_dir or os.path.dirname(pdf_path)
                process_single_pdf(
                    pdf_path,
                    pdf_output_dir,
                    processing_mode,
                    verbose,
                    model,
                    prefer_openrouter,
                )


def process_directory_parallel(
    target_path: str,
    output_dir: Optional[str],
    processing_mode: str,
    verbose: bool,
    model: str,
    prefer_openrouter: bool,
) -> None:
    """
    Process all PDF files in a directory in parallel.
    
    Args:
        target_path: Path to the directory.
        output_dir: Base output directory (None to use same as each PDF).
        processing_mode: Processing mode ('v' or 'vt').
        verbose: Whether to print markdown to console.
        model: Model identifier to use.
        prefer_openrouter: Whether to prefer OpenRouter.
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
                            processing_mode,
                            verbose,
                            model,
                            prefer_openrouter,
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


if __name__ == "__main__":
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
        action="store_true",
        default=False,
        help="If set, process each PDF file in parallel when using recursive "
        "mode.",
    )
    args = parser.parse_args()
    
    # Handle --list-models flag (before importing llm_providers)
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Require target_path if not listing models
    if not args.target_path:
        parser.error("target_path is required unless using --list-models")
    
    # Extract configuration from args
    model = args.model
    prefer_openrouter = not args.prefer_direct
    
    # Print configuration and resolve model
    print_cli_configuration(args, model)
    print_model_resolution(model, prefer_openrouter)
    
    # Process PDF(s) based on mode
    if args.recursive:
        validate_directory_path(args.target_path)
        if args.parallel:
            process_directory_parallel(
                args.target_path,
                args.output_dir,
                args.mode,
                args.verbose,
                model,
                prefer_openrouter,
            )
        else:
            process_directory_sequential(
                args.target_path,
                args.output_dir,
                args.mode,
                args.verbose,
                model,
                prefer_openrouter,
            )
    else:
        validate_file_path(args.target_path)
        output_dir = args.output_dir or os.path.dirname(args.target_path)
        process_single_pdf(
            args.target_path,
            output_dir,
            args.mode,
            args.verbose,
            model,
            prefer_openrouter,
        )

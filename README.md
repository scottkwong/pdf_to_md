
# PDF to Markdown Converter

This tool converts PDF documents to Markdown files using LLM vision models. It supports multiple providers including OpenRouter (primary), OpenAI, Anthropic, and Google. The tool is designed to accurately interpret and transcribe the contents of a PDF, including text and tabular data, into a Markdown format. This script is particularly useful for processing and digitizing documents for easier editing and sharing in a text-based format.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9 or higher
- At least one API key from the supported providers (see [API Keys](#api-keys))
- **poppler-utils** (for PDF-to-image conversion). See [Additional Dependencies](#additional-dependencies) for install commands.

## Installation

Choose your installer. Dependencies are defined in `pyproject.toml`; `requirements.txt` is a legacy fallback.

**Quick start (optional):** From a clone, run `./bootstrap.sh` to create `.venv` and install deps, then `source .venv/bin/activate` and `./pdf_to_md.py --help`.

### pipx (recommended for system-wide CLI)

Install so `pdf_to_md` is on PATH from any directory:

```bash
pipx install git+https://github.com/scottkwong/pdf_to_md.git
```

Optional: add the PyMuPDF parser for faster first-pass text extraction:

```bash
pipx install "git+https://github.com/scottkwong/pdf_to_md.git[pymupdf]"
```

To install a specific version (requires a git tag, e.g. `v0.1.0`):

```bash
pipx install git+https://github.com/scottkwong/pdf_to_md.git@v0.1.0
```

Alternatively, from a local clone:

```bash
cd pdf_to_md
pipx install .
```

Verify:

```bash
pdf_to_md --version
```

### uv (for development / clone-and-run)

```bash
git clone https://github.com/scottkwong/pdf_to_md.git
cd pdf_to_md
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
# Optional: uv pip install -e ".[pymupdf]"
```

### pip

```bash
git clone https://github.com/scottkwong/pdf_to_md.git
cd pdf_to_md
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
# Optional: pip install ".[pymupdf]"
```

If PyMuPDF fails to install on your platform, skip it; the app falls back to pypdf with `--digital-text-parser auto`.

### Conda

```bash
conda create -n pdf_to_md python=3.9
conda activate pdf_to_md
cd pdf_to_md
pip install .
# Optional: pip install ".[pymupdf]"
```

Project dependencies are defined in `pyproject.toml` and installed via pip; Conda is for Python/runtime only.

If you don't have **poppler** installed, see [Additional Dependencies](#additional-dependencies).

### API Keys

You need to configure API keys for the LLM providers you want to use. 
The tool supports two methods:

**Option 1: Create a `.env` file (Recommended)**

Create a `.env` file in the project directory with your API keys:

```bash
# .env file
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_key_here
```

If you have the keys in your shell environment (e.g., in `.bash_profile`
or `.bashrc`), you can use the included helper script:

```bash
source ~/.bash_profile  # or ~/.bashrc
conda activate pdf_to_md
./create_env.sh
```

**Option 2: Set environment variables directly**

Add the API keys to your shell profile (`.bash_profile`, `.bashrc`, or
`.zshrc`):

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
export LLAMA_CLOUD_API_KEY="your_llamacloud_key_here"
```

**Supported API Keys:**
- `OPENROUTER_API_KEY` (recommended - supports all models)
- `OPENAI_API_KEY` (for direct OpenAI access)
- `ANTHROPIC_API_KEY` (for direct Anthropic access)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (for direct Google access)
- `LLAMA_CLOUD_API_KEY` (for LlamaParse extraction - get from https://cloud.llamaindex.ai)

**Note:** You only need at least one API key to use the tool. OpenRouter
is recommended as it provides access to all supported models through a
single API. For LlamaParse extraction, you need a separate `LLAMA_CLOUD_API_KEY`.

The tool defaults to using OpenRouter when available, with automatic 
fallback to direct provider APIs. If your requested model is unavailable, 
you'll be prompted to select from available alternatives.


## Virtual environment (venv or Conda)

For development or clone-and-run, use a virtual environment in the repo (e.g. `.venv`) or a Conda env. Activate it before running the script or installing dependencies. If you use `.venv`, Cursor can pick it automatically via `.vscode/settings.json`.

**Configuring Cursor/VS Code:** Use **Python: Select Interpreter** (`Cmd+Shift+P` / `Ctrl+Shift+P`) and choose the interpreter of the environment you use (e.g. `.venv/bin/python` or your Conda env’s `python`). The bottom-right corner shows the active interpreter.

## Running the tool

**From the project directory:** Activate your environment, then run:

```bash
./pdf_to_md.py <path_to_pdf>
# or
python pdf_to_md.py <path_to_pdf>
# or (if installed in the env)
python -m pdf_to_md <path_to_pdf>
```

**From another folder (global command):** Prefer **pipx install from Git** (see [pipx](#pipx-recommended-for-system-wide-cli) above) so `pdf_to_md` is on PATH everywhere. Check which version is running with `pdf_to_md --version`.

If you don’t use pipx, you can use a small wrapper script that invokes the repo’s environment: create a script (e.g. `~/bin/pdf_to_md`) that runs:

```bash
exec "/absolute/path/to/pdf_to_md/.venv/bin/python" "/absolute/path/to/pdf_to_md/pdf_to_md.py" "$@"
```

Then `chmod +x ~/bin/pdf_to_md` and ensure that directory is in your `PATH`. You can pass any PDF path, e.g. `pdf_to_md ~/Documents/foo.pdf`.

## Usage

Convert PDF files to Markdown format using LLM vision models. It supports processing a single file or multiple files within a directory, optionally in parallel. The script provides several options including model selection, provider choice, output directory specification, processing modes, verbosity, and recursive directory processing.

The PDF pipeline is:

1. Extract per-page first-pass text with a digital text parser
   (`pypdf` or `pymupdf`).
2. If mode is `vt`, pass page image + first-pass text to the LLM provider.
3. If mode is `v`, pass only the page image to the LLM provider.

### Basic Usage

To convert a PDF file (defaults to OpenAI GPT-5.2 via OpenRouter):

```bash
./pdf_to_md.py <path_to_pdf>
```

### Advanced Usage

To utilize additional options:

```bash
./pdf_to_md.py <path_to_pdf> -o <output_directory> -m <mode> --model <model> --provider <provider> --prefer-direct -q -r -p
```

**Options:**

- `<path_to_pdf>`: Path to the PDF file or directory containing PDF files.
- `--list-models`: List all available models and show the default, then exit.
- `--version`: Show version and exit.
- `-o`, `--output_dir <output_directory>`: Destination for Markdown files. Defaults to PDF's location if unspecified.
- `-m`, `--mode <mode>`: Sets processing mode. Choose 'v' for vision-only or 'vt' for vision-and-text (default: 'vt').
- `--digital-text-parser <parser>`: Parser engine used for first-pass digital text parsing in `vt` mode. Options: `auto` (default), `pypdf`, `pymupdf`.
- `--model <model>`: Model identifier from `models.json` to use for both vision and text processing (default: `gpt-5.2`).
- `--provider <provider>`: Force specific provider: `openrouter`, `openai`, `anthropic`, or `google` (optional).
- `--prefer-direct`: Skip OpenRouter and use direct APIs only.
- `-q`, `--quiet`: Disables verbose output. By default, the script prints markdown to console.
- `-r`, `--recursive`: Processes all PDF files within the target directory recursively. Files are processed in parallel by default.
- `-s`, `--single`: Process files sequentially instead of in parallel when using `-r`.
- `-p`, `--parallel [N]`: Max parallel pages to process with VisionExtractor (default: 10). Controls how many pages are processed concurrently.
- `-d`, `--debug`: Enable debug logging for page processing order. Useful for diagnosing page ordering issues.
- `--extractor <extractor>`: Extraction method: `vision` (default, LLM-based) or `llamaparse` (LlamaCloud API).
- `--llamaparse-tier <tier>`: LlamaParse processing tier (only with `--extractor llamaparse`): `fast`, `cost_effective`, `agentic` (default), `agentic_plus`.
- `--language <lang>`: Document language code for LlamaParse (default: `en`).
- `--benchmark-digital-text-parsers`: Run a direct package benchmark (no LLM calls) comparing pypdf and PyMuPDF digital text parsers, then exit.
- `--benchmark-runs <N>`: Number of benchmark runs per parser (default: `10`).
- `--benchmark-pdf <path>`: Optional PDF override for benchmark mode. If omitted, benchmark uses generated fixture PDFs from `tests/fixtures`.

**Available Models** (defined in `models.json`):
- `gpt-5.2` - OpenAI GPT-5.2 (default)
- `openai-gpt4o` - OpenAI GPT-4o
- `gemini-3-flash` - Google Gemini 3 Flash
- `gemini-3-pro` - Google Gemini 3 Pro
- `claude-sonnet-4.5` - Anthropic Claude Sonnet 4.5
- `claude-opus-4.5` - Anthropic Claude Opus 4.5
- `claude-opus-4.6` - Anthropic Claude Opus 4.6
- `claude-haiku-4.5` - Anthropic Claude Haiku 4.5

**Examples:**

```bash
# List all available models
./pdf_to_md.py --list-models

# Use default model (GPT-5.2 via OpenRouter)
./pdf_to_md.py document.pdf

# Use Gemini 3 Pro
./pdf_to_md.py document.pdf --model gemini-3-pro

# Use Claude Sonnet with direct API (skip OpenRouter)
./pdf_to_md.py document.pdf --model claude-sonnet-4.5 --prefer-direct

# Use specific provider
./pdf_to_md.py document.pdf --model claude-sonnet-4.5 --provider anthropic

# Process directory recursively (files processed in parallel by default)
./pdf_to_md.py ./documents -r

# Process directory sequentially (one file at a time)
./pdf_to_md.py ./documents -r -s

# Process directory quietly (no console output)
./pdf_to_md.py ./documents -r -q

# Limit parallel page processing to 5 concurrent pages
./pdf_to_md.py document.pdf -p 5

# Enable debug logging to see page processing order
./pdf_to_md.py document.pdf -d

# Use LlamaParse extraction (requires LLAMA_CLOUD_API_KEY)
./pdf_to_md.py document.pdf --extractor llamaparse

# Use LlamaParse with maximum fidelity tier
./pdf_to_md.py document.pdf --extractor llamaparse --llamaparse-tier agentic_plus

# Use LlamaParse for non-English documents
./pdf_to_md.py document.pdf --extractor llamaparse --language de

# Force pypdf digital text parser
./pdf_to_md.py document.pdf --digital-text-parser pypdf

# Run digital text parser benchmark against generated fixture PDFs (default path)
./pdf_to_md.py --benchmark-digital-text-parsers --benchmark-runs 10

# Run digital text parser benchmark against a specific PDF
./pdf_to_md.py --benchmark-digital-text-parsers --benchmark-pdf document.pdf
```

### Extraction Methods

The tool supports two extraction methods:

**Vision-based extraction (default):** Converts PDF pages to images and uses
vision-capable LLMs (GPT-4, Claude, Gemini) to interpret each page. Best for
documents where visual layout is important.

When mode is `vt` (default), vision extraction includes first-pass text from a
digital text parser:
- `auto` (default): prefers PyMuPDF when installed, otherwise pypdf
- `pypdf`: forces pypdf backend
- `pymupdf`: forces PyMuPDF backend (requires `pip install ".[pymupdf]"` or `pip install pymupdf`)

**LlamaParse extraction:** Uses LlamaIndex's LlamaParse API for document-level
extraction. Optimized for structured documents like financial reports and
scientific papers. Requires `LLAMA_CLOUD_API_KEY`.

LlamaParse tiers:
- `fast` - Speed priority, best for simple documents
- `cost_effective` - Budget-friendly for standard documents
- `agentic` - Balanced accuracy and speed (default)
- `agentic_plus` - Maximum fidelity for complex layouts

LlamaParse pricing: Free tier includes 1,000 pages/day. Paid plans offer
7,000 pages/week + $0.003 per additional page.

### Provider Selection

The tool automatically selects the best provider based on available API keys:

1. **OpenRouter** (if `OPENROUTER_API_KEY` is set) - Supports all models
2. **Direct Provider APIs** - Falls back to direct APIs if OpenRouter unavailable
3. **Interactive Fallback** - If requested model is unavailable, you'll be prompted to select from available alternatives

## Testing

Run the test suite to verify all providers work correctly:

```bash
python run_tests.py
```

This will:
- Check which API keys are available
- Run text processing tests for each available provider
- Run vision processing tests for each available provider
- Provide a summary of results

Tests use your `.env` file for API keys (no hardcoded values).

### Digital text parser and setup tests (no LLM calls)

These tests validate direct parser correctness and installation behavior:

```bash
python -m pytest tests/test_digital_text_parsers.py
python -m pytest tests/test_installation_setup.py
python -m pytest tests/test_benchmark_digital_text_parsers.py
```

### Digital text parser benchmark

Benchmark direct package extraction with mean and standard deviation:

```bash
# Default: generated fixture PDFs in tests/fixtures
python pdf_to_md.py --benchmark-digital-text-parsers --benchmark-runs 10

# Override input PDF
python pdf_to_md.py --benchmark-digital-text-parsers --benchmark-pdf path/to/file.pdf
```


## License

This project is open source and available under the [MIT License](LICENSE.txt).


## Additional Dependencies

Aside from the Python packages in `pyproject.toml` (or `requirements.txt`), this project requires `poppler-utils` to be installed on your system. `poppler-utils` includes utilities like `pdftoppm` which are essential for PDF processing.

### Installing poppler-utils

#### On Ubuntu/Debian-based Linux Distributions:

Run the following command in your terminal:

```bash
sudo apt-get install -y poppler-utils
```

#### On macOS:

If you have Homebrew installed, you can run:

```bash
brew install poppler
```

If you do not have Homebrew, you can install it from [here](https://brew.sh/).

### Verifying the Installation

To ensure that `poppler-utils` has been installed correctly, you can run:

```bash
pdftoppm -v
```

This command should return the version of `pdftoppm` if `poppler-utils` is installed correctly.

### Optional: PyMuPDF

PyMuPDF is optional and can be used as the first-pass digital text parser. Install via the extra (from the project directory) or standalone:

```bash
pip install ".[pymupdf]"
# or: pip install pymupdf
# or with uv: uv pip install -e ".[pymupdf]"
```

Verification:

```bash
python -c "import fitz; print(fitz.__doc__.splitlines()[0])"
```

If installation fails due to architecture-specific wheel/build issues, use
pypdf only:

```bash
./pdf_to_md.py document.pdf --digital-text-parser pypdf
```

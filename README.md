
# PDF to Markdown Converter

This tool converts PDF documents to Markdown files using LLM vision models. It supports multiple providers including OpenRouter (primary), OpenAI, Anthropic, and Google. The tool is designed to accurately interpret and transcribe the contents of a PDF, including text and tabular data, into a Markdown format. This script is particularly useful for processing and digitizing documents for easier editing and sharing in a text-based format.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9 or higher
- Pip (Python package installer)
- At least one API key from the supported providers (see [API Keys](#api-keys))

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/scottkwong/pdf-to-md.git
cd pdf_to_md
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you don't have `poppler` installed, see [Additional Dependencies](#additional-dependencies).

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
```

**Supported API Keys:**
- `OPENROUTER_API_KEY` (recommended - supports all models)
- `OPENAI_API_KEY` (for direct OpenAI access)
- `ANTHROPIC_API_KEY` (for direct Anthropic access)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (for direct Google access)

**Note:** You only need at least one API key to use the tool. OpenRouter 
is recommended as it provides access to all supported models through a 
single API.

The tool defaults to using OpenRouter when available, with automatic 
fallback to direct provider APIs. If your requested model is unavailable, 
you'll be prompted to select from available alternatives.


## Virtual Environment Setup (Recommended)

It's recommended to use a Conda environment to manage dependencies for this project.

### Creating the Conda Environment

Create a new Conda environment named `pdf_to_md` with Python 3.9 or higher:

```bash
conda create -n pdf_to_md python=3.9
```

**Note:** Python 3.9+ is required for the `anthropic` and `google-genai` packages.

### Activating the Environment

Before running the script or installing dependencies, activate the environment:

```bash
conda activate pdf_to_md
```

Once activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Deactivating the Environment

When you're done working with the script, you can deactivate the environment:

```bash
conda deactivate
```

### Checking Active Environment

To verify which environment is currently active:

```bash
conda env list
```

The active environment will be marked with an asterisk (*).

### Configuring Cursor/VS Code

To ensure Cursor uses the correct Conda environment for linting and
IntelliSense:

1. **Automatic (recommended):** A `.vscode/settings.json` file is included
   that points to the `pdf_to_md` Conda environment.

2. **Manual selection:** 
   - Press `Cmd+Shift+P` (or `Ctrl+Shift+P` on Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose `/opt/homebrew/anaconda3/envs/pdf_to_md/bin/python`

3. **Verify:** Check the bottom-right corner of Cursor to confirm it shows
   "3.x.x ('pdf_to_md')"

If you installed Anaconda in a different location, update the path in
`.vscode/settings.json` accordingly.


## Script Configuration

Before running `pdf_to_md.py`, ensure the shebang line (first line in the file) points to your Python interpreter. If needed, replace `#!/opt/homebrew/anaconda3/envs/pdf_to_md/bin/python` with the path to your Python executable, which you can find with `which python` or `which python3` in your terminal.


## Usage

Convert PDF files to Markdown format using LLM vision models. It supports processing a single file or multiple files within a directory, optionally in parallel. The script provides several options including model selection, provider choice, output directory specification, processing modes, verbosity, and recursive directory processing.

### Basic Usage

To convert a PDF file (defaults to OpenAI GPT-5.2 via OpenRouter):

```bash
./pdf_to_md.py <path_to_pdf>
```

### Advanced Usage

To utilize additional options:

```bash
./pdf_to_md.py <path_to_pdf> -o <output_directory> -m <mode> --vision-model <model> -v -r -p
```

**Options:**

- `<path_to_pdf>`: Path to the PDF file or directory containing PDF files.
- `-o`, `--output_dir <output_directory>`: Destination for Markdown files. Defaults to PDF's location if unspecified.
- `-m`, `--mode <mode>`: Sets processing mode. Choose 'v' for vision-only or 'vt' for vision-and-text (default: 'vt').
- `--vision-model <model>`: Model identifier from `models.json` for vision processing (default: `gpt-5.2`).
- `--text-model <model>`: Model identifier for text processing (defaults to vision-model).
- `--provider <provider>`: Force specific provider: `openrouter`, `openai`, `anthropic`, or `google` (optional).
- `--prefer-direct`: Skip OpenRouter and use direct APIs only.
- `-v`, `--verbose`: Enables verbose output, printing the Markdown text to the console.
- `-q`, `--quiet`: Disables verbose output (opposite of `-v`).
- `-r`, `--recursive`: Processes all PDF files within the target directory recursively.
- `-p`, `--parallel`: Processes files in parallel during recursive operation.

**Available Models** (defined in `models.json`):
- `gpt-5.2` - OpenAI GPT-5.2 (default)
- `openai-gpt4o` - OpenAI GPT-4o
- `gemini-3-flash` - Google Gemini 3 Flash
- `gemini-3-pro` - Google Gemini 3 Pro
- `claude-sonnet-4.5` - Anthropic Claude 3.5 Sonnet
- `claude-opus-4.5` - Anthropic Claude 3.5 Opus
- `claude-haiku-4.5` - Anthropic Claude 3.5 Haiku

**Examples:**

```bash
# Use default model (GPT-5.2 via OpenRouter)
./pdf_to_md.py document.pdf

# Use Gemini 3 Pro
./pdf_to_md.py document.pdf --vision-model gemini-3-pro

# Use Claude Sonnet with direct API (skip OpenRouter)
./pdf_to_md.py document.pdf --vision-model claude-sonnet-4.5 --prefer-direct

# Process directory recursively with verbose output
./pdf_to_md.py ./documents -r -v
```

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


## License

This project is open source and available under the [MIT License](LICENSE.txt).


## Additional Dependencies

Aside from the Python packages listed in `requirements.txt`, this project requires `poppler-utils` to be installed on your system. `poppler-utils` includes utilities like `pdftoppm` which are essential for PDF processing.

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

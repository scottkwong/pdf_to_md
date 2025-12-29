
# PDF to Markdown Converter

This tool converts PDF documents to Markdown files using GPT-4's visual reasoning capabilities. It's designed to accurately interpret and transcribe the contents of a PDF, including text and tabular data, into a Markdown format. This script is particularly useful for processing and digitizing documents for easier editing and sharing in a text-based format.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3
- Pip (Python package installer)
- [An OpenAI API key](https://beta.openai.com/signup/) for GPT-4 access

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

Set your OpenAI API key in your environment or `.env` file as `OPENAI_API_KEY`.


## Virtual Environment Setup (Recommended)

It's recommended to use a Conda environment to manage dependencies for this project.

### Creating the Conda Environment

Create a new Conda environment named `pdf_to_md`:

```bash
conda create -n pdf_to_md python=3.9
```

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

Convert PDF files to Markdown format using GPT-4's visual reasoning. It supports processing a single file or multiple files within a directory, optionally in parallel. The script provides several options, including output directory specification, processing modes, verbosity for detailed output, and recursive directory processing.

### Basic Usage

To convert a PDF file:

```bash
./pdf_to_md.py <path_to_pdf>
```

### Advanced Usage

To utilize additional options:

```bash
./pdf_to_md.py <path_to_pdf> -o <output_directory> -m <mode> -v -r -p
```

Options:

- `<path_to_pdf>`: Path to the PDF file or directory containing PDF files.
- `-o`, `--output_dir <output_directory>`: Destination for Markdown files. Defaults to PDF's location if unspecified.
- `-m`, `--mode <mode>`: Sets processing mode. Choose 'v' for vision-only or 'vt' for vision-and-text (default: 'vt').
- `-v`, `--verbose`: Enables verbose output, printing the Markdown text to the console.
- `-r`, `--recursive`: Processes all PDF files within the target directory recursively.
- `-p`, `--parallel`: Processes files in parallel during recursive operation.

Ensure to replace `<path_to_pdf>` and `<output_directory>` with your specific paths. The `-m` option allows for tailored processing, while `-v`, `-r`, and `-p` flags offer control over output verbosity, directory traversal, and execution strategy, respectively.


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


# PDF to Markdown Converter

This tool converts PDF documents to Markdown files using OpenAI's visual reasoning capabilities. It's designed to accurately interpret and transcribe the contents of a PDF, including text and tabular data, into a Markdown format. This script is particularly useful for processing and digitizing documents for easier editing and sharing in a text-based format.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3
- Pip (Python package installer)
- [An OpenAI API key](https://beta.openai.com/signup/) for model access

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


## Script Configuration

Before running `pdf_to_md.py`, ensure the shebang line (first line in the file) points to your Python interpreter. If needed, replace `#!/opt/homebrew/anaconda3/envs/pdf_to_md/bin/python` with the path to your Python executable, which you can find with `which python` or `which python3` in your terminal.


## Usage

Convert PDF files to Markdown format using OpenAI's visual reasoning models. It supports processing a single file or multiple files within a directory, optionally in parallel. The script provides several options, including output directory specification, processing modes, verbosity for detailed output, model selection, and recursive directory processing.

### Basic Usage

To convert a PDF file:

```bash
./pdf_to_md.py <path_to_pdf>
```

### Advanced Usage

To utilize additional options:

```bash
./pdf_to_md.py <path_to_pdf> -o <output_directory> -m <mode> --model gpt-4o -v -r -p
```

Options:

- `<path_to_pdf>`: Path to the PDF file or directory containing PDF files.
- `-o`, `--output_dir <output_directory>`: Destination for Markdown files. Defaults to PDF's location if unspecified.
- `-m`, `--mode <mode>`: Sets processing mode. Choose 'v' for vision-only or 'vt' for vision-and-text (default: 'vt').
- `--model <model>`: Selects the OpenAI model to use (default: `gpt-4o`).
- `-v`, `--verbose`: Enables verbose output, printing the Markdown text to the console.
- `-r`, `--recursive`: Processes all PDF files within the target directory recursively.
- `-p`, `--parallel`: Processes files in parallel during recursive operation.

Ensure to replace `<path_to_pdf>` and `<output_directory>` with your specific paths. The `-m` option allows for tailored processing, and `--model` chooses the model variant. The `-v`, `-r`, and `-p` flags offer control over output verbosity, directory traversal, and execution strategy, respectively.


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

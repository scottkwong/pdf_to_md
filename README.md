
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
pipx install "pdf-to-md @ git+https://github.com/scottkwong/pdf_to_md.git"
```

Optional: add the PyMuPDF parser for faster first-pass text extraction:

```bash
pipx install "pdf-to-md[pymupdf] @ git+https://github.com/scottkwong/pdf_to_md.git"
```

To install a specific version (requires a git tag, e.g. `v0.1.0`):

```bash
pipx install "pdf-to-md @ git+https://github.com/scottkwong/pdf_to_md.git@v0.1.0"
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

To convert a PDF file (defaults to OpenAI GPT-5.4 via OpenRouter):

```bash
./pdf_to_md.py <path_to_pdf>
```

### Advanced Usage

To utilize additional options:

```bash
./pdf_to_md.py <path_to_pdf> -o <output_directory> -m <mode> --model <model> --provider <provider> --prefer-openrouter -q -r -p
```

**Options:**

- `<path_to_pdf>`: Path to the PDF file or directory containing PDF files.
- `--list-models`: List all available models and show the default, then exit.
- `--version`: Show version and exit.
- `-o`, `--output_dir <output_directory>`: Destination for Markdown files. Defaults to PDF's location if unspecified.
- `-m`, `--mode <mode>`: Sets processing mode. Choose 'v' for vision-only or 'vt' for vision-and-text (default: 'vt').
- `--digital-text-parser <parser>`: Parser engine used for first-pass digital text parsing in `vt` mode. Options: `auto` (default), `pypdf`, `pymupdf`.
- `--model <model>`: Model identifier from `models.json` to use for both vision and text processing (default: `gpt-5.5`).
- `--provider <provider>`: Force specific provider: `openrouter`, `openai`, `anthropic`, or `google` (optional).
- `--prefer-openrouter`: Route via OpenRouter when available. Default is to use direct provider APIs.
- `-q`, `--quiet`: Disables verbose output. By default, the script prints markdown to console.
- `-r`, `--recursive`: Processes all PDF files within the target directory recursively. Files are processed in parallel by default.
- `-s`, `--single`: Process files sequentially instead of in parallel when using `-r`.
- `-p`, `--parallel [N]`: Max parallel pages to process with VisionExtractor (default: 10). Controls how many pages are processed concurrently.
- `-d`, `--debug`: Enable debug logging for page processing order. Useful for diagnosing page ordering issues.
- `--extractor <extractor>`: Extraction method: `vision` (default, LLM-based) or `llamaparse` (LlamaCloud API). Ignored when `--local` is set.
- `--local`: Run OCR fully offline against a local model server. No API keys and no per-token cost.
- `--local-backend <backend>`: Local runtime used with `--local`: `ollama` (default), `vllm` (an OpenAI-compatible vLLM server, e.g. for [KDL-Frontier-Parser-nano](https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano)), or `unlimited-ocr` (a vLLM server serving [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)).
- `--local-model <tag>`: Ollama vision model tag to use with `--local --local-backend ollama` and as the local entry in `--benchmark` (default: `qwen2.5vl:7b`).
- `--ollama-url <url>`: Ollama generate endpoint for `--local --local-backend ollama` / local benchmark models (default: `http://localhost:11434/api/generate`).
- `--vllm-model <name>`: Served-model name to request from the vLLM server with `--local --local-backend vllm` (default: `kdl-frontier-parser-nano`).
- `--vllm-url <url>`: vLLM OpenAI-compatible base URL for `--local --local-backend vllm` / `vllm` benchmark entries (default: `http://localhost:8000/v1`).
- `--vllm-max-tokens <N>`: Output token budget per page for `--local --local-backend vllm` / `vllm` benchmark entries (default: `4096`). Keep it below the server's `--max-model-len` minus the page image and prompt.
- `--unlimited-model <name>`: Served-model name to request from the vLLM server with `--local --local-backend unlimited-ocr` / `unlimited` benchmark entries (default: `unlimited-ocr`).
- `--unlimited-url <url>`: vLLM OpenAI-compatible base URL for Baidu Unlimited-OCR (default: `http://localhost:8001/v1`). A separate model needs a separate server, so it defaults to a different port than `--vllm-url`.
- `--unlimited-max-tokens <N>`: Output token budget per page for `--local --local-backend unlimited-ocr` / `unlimited` benchmark entries (default: `8192`).
- `--llamaparse-tier <tier>`: LlamaParse processing tier (only with `--extractor llamaparse`): `fast`, `cost_effective`, `agentic` (default), `agentic_plus`.
- `--language <lang>`: Document language code for LlamaParse (default: `en`).
- `--benchmark`: Benchmark and compare the top model from each provider (OpenAI, Anthropic, and Local) on the same PDF, then exit. Runs models in parallel and streams progress. The local model may download weights on first run.
- `--benchmark-models <list>`: Comma-separated override for `--benchmark`, e.g. `gpt-5.5,claude-opus-4.6,local`. Use `local` or `local:<model>` for an Ollama model, `vllm` or `vllm:<model>` for a local vLLM served model (e.g. KDL-Frontier-Parser-nano), `unlimited` or `unlimited:<model>` for a local Baidu Unlimited-OCR server; other entries are `models.json` keys.
- `--benchmark-reference <label>`: Model label used as the baseline in the `--benchmark` HTML comparison; every other model is diffed against it and it is shown first. Defaults to the first benchmarked model. Remaining models are ordered cheapest-first.
- `--benchmark-digital-text-parsers`: Run a direct package benchmark (no LLM calls) comparing pypdf and PyMuPDF digital text parsers, then exit.
- `--benchmark-runs <N>`: Number of benchmark runs per parser (default: `10`).
- `--benchmark-pdf <path>`: Optional PDF override for benchmark modes. If omitted, `--benchmark-digital-text-parsers` and `--benchmark` use generated fixture PDFs from `tests/fixtures`.

**Available Models** (defined in `models.json`):
- `gpt-5.5` - OpenAI GPT-5.5 (default)
- `gpt-5.5-pro` - OpenAI GPT-5.5 Pro
- `gpt-5.4` - OpenAI GPT-5.4
- `gpt-5.2` - OpenAI GPT-5.2
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

# Use default model (GPT-5.5 via direct OpenAI API)
./pdf_to_md.py document.pdf

# Use Gemini 3 Pro
./pdf_to_md.py document.pdf --model gemini-3-pro

# Route via OpenRouter instead of direct provider APIs
./pdf_to_md.py document.pdf --model claude-sonnet-4.5 --prefer-openrouter

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

# Run fully offline against a local Ollama server (no API keys)
./pdf_to_md.py document.pdf --local

# Use a different local model
./pdf_to_md.py document.pdf --local --local-model granite3.2-vision

# Point at an Ollama server on another host
./pdf_to_md.py document.pdf --local --ollama-url http://192.168.1.10:11434/api/generate

# Run offline against a local vLLM server serving KDL-Frontier-Parser-nano
./pdf_to_md.py document.pdf --local --local-backend vllm

# Point at a vLLM server on another host / port
./pdf_to_md.py document.pdf --local --local-backend vllm --vllm-url http://192.168.1.10:8000/v1

# Run offline against a local vLLM server serving Baidu Unlimited-OCR
./pdf_to_md.py document.pdf --local --local-backend unlimited-ocr

# Benchmark the top OpenAI, Anthropic, and Local models on one PDF (parallel)
./pdf_to_md.py document.pdf --benchmark

# Benchmark a custom set of models
./pdf_to_md.py --benchmark --benchmark-models "gpt-5.5,claude-opus-4.6,local:llava" --benchmark-pdf document.pdf

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

**Local extraction (`--local`):** Runs OCR fully offline against a local model
server. No API keys and no per-token cost — your documents never leave the
machine. It reuses the same page-by-page parallel pipeline as vision extraction
(including `vt` first-pass digital text), just swapping the cloud provider for a
local model. Two backends are available, selected with `--local-backend`:

- `ollama` (default) — a local [Ollama](https://ollama.com) server running a
  general vision model (Qwen-VL, LLaVA, Granite Vision, …). Talks to Ollama's
  HTTP API directly; no extra Python dependency.
- `vllm` — a local [vLLM](https://docs.vllm.ai) server exposing an
  OpenAI-compatible API, for purpose-built document-parsing models such as
  [KDL-Frontier-Parser-nano](https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano).
  Uses the bundled `openai` client; no extra Python dependency. See
  [vLLM backend](#vllm-backend-kdl-frontier-parser) below.

#### Ollama backend (default)

Setup:

```bash
# 1. Install Ollama (https://ollama.com) and start the server
ollama serve

# 2. Run offline (the model is pulled automatically on first use)
./pdf_to_md.py document.pdf --local
```

The first run downloads the model weights (`qwen2.5vl:7b` by default, ~6 GB);
progress is streamed to the console. Choose another Ollama vision model with
`--local-model` and point at a non-default server with `--ollama-url`.

**Choosing a local model.** Pick one built for transcription, and verify it on
your own documents before trusting it:

- Caption-oriented models (e.g. `moondream`) describe a page instead of
  transcribing it, and produce unusable output.
- Reasoning ("thinking") models such as `qwen3-vl` work, but spend most of the
  per-page output budget on reasoning before answering, so they are several
  times slower. Thinking cannot be turned off — Ollama accepts `think: false`
  and a `/no_think` prompt but the model reasons anyway. If one exhausts its
  budget without answering, the page fails with a message naming the flag to
  raise, rather than landing in the document as a silently blank page.
- `llama3.2-vision` no longer loads on current Ollama releases, which dropped
  its `mllama` architecture.

**Context and output budget.** Both are sized per request, for one page:

- `--local-num-ctx` (default 16384) — the context window. We send ~4-5k tokens
  per page (page image + `vt` prior text + prompt). This is set explicitly
  because some vision models default to a very large context: `qwen3-vl`
  defaults to 262k, whose KV cache alone reserves tens of GB and can push a
  64 GB machine into swap. At 16384 the same model loads in ~8 GB.
- `--local-num-predict` (default 8192) — output tokens per page. A
  non-reasoning model stops well short of this and pays nothing for the
  headroom. A reasoning model spends it thinking first, so raise it (and
  `--local-num-ctx` to fit it) if pages fail on their budget.

`num_ctx` must accommodate the input *plus* `num_predict`, so raise them
together.

#### vLLM backend (KDL-Frontier-Parser)

`--local-backend vllm` targets purpose-built document-parsing VLMs that ship on
Hugging Face and are served with [vLLM](https://docs.vllm.ai) rather than
Ollama — in particular
[KDL-Frontier-Parser-nano](https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano),
a 1.2B Qwen2-VL-based parser (MinerU 2.5 lineage) tuned for OCR, tables, and
charts. Like the Ollama backend it plugs into the same parallel page pipeline;
the app calls the server's OpenAI-compatible `/v1/chat/completions` endpoint
once per page and never downloads or launches the model itself — the vLLM
server is started out of band and owns its own GPU memory and model loading.

Setup:

```bash
# 1. Install vLLM (see https://docs.vllm.ai) — needs a CUDA GPU.
pip install vllm

# 2. Serve the model (per the model card). The served-model name must match
#    --vllm-model below (default: kdl-frontier-parser-nano).
vllm serve KDLAI/KDL-Frontier-Parser-nano \
  --served-model-name kdl-frontier-parser-nano \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 24 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":1}'

# 3. Convert offline against the running server.
./pdf_to_md.py document.pdf --local --local-backend vllm
```

Point at a non-default server with `--vllm-url` (default
`http://localhost:8000/v1`), request a different served-model name with
`--vllm-model`, and cap the per-page output with `--vllm-max-tokens` (default
4096; keep it below the server's `--max-model-len` minus the page image and
prompt). If the server is unreachable, or is serving a different model name, the
run stops before any page is processed with a message that names the fix. Each
page is a single end-to-end pass with greedy decoding, and the model-card
requirements (`enable_thinking=False`, `skip_special_tokens=False`) are set on
every request. If the local server is started with `--api-key`, export the same
value as `VLLM_API_KEY`.

#### Unlimited-OCR backend (Baidu)

`--local-backend unlimited-ocr` targets
[Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), a 3.3B
DeepSeek-OCR-lineage model for long documents. It is also served with vLLM, but
its recipe differs from KDL's and the app handles the differences for you:

- It has **no chat template** and is trained on a fixed prompt that must begin
  with `<image>`, so the app sends that recipe prompt (ignoring the generic
  markdown prompt and any `vt` prior text).
- Its raw output carries `<|ref|>…<|/ref|>` / `<|det|>…<|/det|>` grounding
  markup; the app strips the coordinate boxes and unwraps the reference spans to
  leave clean markdown.
- The server needs the model's **no-repeat-ngram logits processor** registered
  at launch, or long pages loop on coordinate tokens.

Because a vLLM server serves one model, Unlimited-OCR runs as its **own server**
on its **own port** — it defaults to `http://localhost:8001/v1`, separate from
KDL's `:8000`.

Setup:

```bash
# 1. Serve Unlimited-OCR on its own port (8001). Its architecture ships in a
#    dedicated vLLM release image; register the required logits processor. See
#    the model card for the exact image tag.
vllm serve baidu/Unlimited-OCR \
  --served-model-name unlimited-ocr \
  --port 8001 \
  --max-model-len 32768 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":1}' \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor

# 2. Convert offline against it.
./pdf_to_md.py document.pdf --local --local-backend unlimited-ocr
```

Point at a different server with `--unlimited-url`, name a different served
model with `--unlimited-model`, and cap per-page output with
`--unlimited-max-tokens` (default 8192). As with the KDL backend, an unreachable
or mis-named server stops the run before any page is processed.

### Model benchmark (`--benchmark`)

`--benchmark` runs the **same PDF** through the top model from each provider
and prints a side-by-side comparison of time, pages, output size, and cost.
By default it compares:

- **OpenAI** — `gpt-5.5` (requires `OPENAI_API_KEY`)
- **Anthropic** — `claude-opus-4.6` (requires `ANTHROPIC_API_KEY`)
- **Local** — `qwen2.5vl:7b` via a local Ollama server

Models run **concurrently** (one worker per model), and each model's own
extractor processes its pages in parallel, so the benchmark does as much work
at once as possible. Progress for every model is streamed with a per-model
`[label]` tag, and the local model's weight download (if needed) is streamed
as progress too. Any model whose API key is missing, or whose Ollama server is
unreachable, is reported as **skipped** with the reason rather than failing the
run.

```bash
# Compare top OpenAI, Anthropic, and Local models on one PDF
./pdf_to_md.py document.pdf --benchmark

# No PDF given -> uses a generated fixture PDF
./pdf_to_md.py --benchmark

# Custom set; write artifacts to ./out instead of ./benchmark_output
./pdf_to_md.py --benchmark \
  --benchmark-models "gpt-5.5,claude-opus-4.6,local" \
  --benchmark-pdf document.pdf -o ./out

# Include a local vLLM model (e.g. KDL-Frontier-Parser-nano) in the comparison
./pdf_to_md.py --benchmark \
  --benchmark-models "gpt-5.5,local,vllm" \
  --benchmark-pdf document.pdf
```

#### Compare the local models head-to-head (Qwen vs KDL vs Unlimited-OCR)

To try the two Hugging Face parsers against the Qwen baseline on your own PDF —
all offline, no API keys — start the three local servers, then run one command.
On one GPU the two vLLM servers can't both hold weights at full utilization, so
lower `--gpu-memory-utilization` on each (or run the pairwise benchmarks in
`--benchmark-models` separately).

```bash
# 1. Ollama for Qwen (the "local" entry)
ollama serve                       # in one terminal

# 2. KDL-Frontier-Parser-nano on :8000 (the "vllm" entry)
vllm serve KDLAI/KDL-Frontier-Parser-nano \
  --served-model-name kdl-frontier-parser-nano --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.45 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}'

# 3. Baidu Unlimited-OCR on :8001 (the "unlimited" entry)
vllm serve baidu/Unlimited-OCR \
  --served-model-name unlimited-ocr --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.45 \
  --trust-remote-code --limit-mm-per-prompt '{"image":1}' \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor

# 4. Benchmark all three on your PDF (writes markdown + an HTML comparison).
./pdf_to_md.py --benchmark \
  --benchmark-models "local,vllm,unlimited" \
  --benchmark-pdf document.pdf
```

Any server that isn't up is reported as **skipped** with the reason, so you can
also start just one of the two parsers and compare it against Qwen. The
local models run one at a time (they share the GPU) while the comparison HTML
lets you judge quality page by page.

#### Benchmark artifacts

Every benchmark run lands its artifacts in `-o <dir>`, or `./benchmark_output`
when `-o` is omitted:

- `<pdf>.<model>.md` — each model's full markdown, one file per model.
- `<pdf>.comparison.html` — a scrollable visual comparison.

Open the HTML to judge quality rather than infer it from the summary table. For
every page it shows the rendered PDF page beside each model's markdown for that
page, so you can read what each model saw against what it produced:

- The **reference** model leads and is the baseline. Every other model is
  word-diffed against it: <ins>green</ins> is text only that model produced,
  <del>red</del> is reference text it missed. Remaining models follow
  cheapest-first.
- Choose the baseline with `--benchmark-reference "<label>"` (use the label
  exactly as it appears in the summary table, e.g. `Local (qwen2.5vl:7b)`).
  Defaults to the first benchmarked model.
- Diffs compare **content, not formatting**, so `**Total**` and `Total` do not
  register as a difference and the highlighting stays readable.

The file is self-contained (page images and CSS are inlined), so it can be
moved or shared as-is.

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

Tests use your `.env` file for API keys.

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

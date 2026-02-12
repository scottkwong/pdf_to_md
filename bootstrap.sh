#!/usr/bin/env bash
# Create .venv and install project dependencies (required + optional pymupdf).
# Run from the repo root. Afterward: source .venv/bin/activate

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  echo ".venv already exists; skipping creation."
else
  python3 -m venv .venv
  echo "Created .venv"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if command -v uv &>/dev/null; then
  uv pip install -e .
else
  pip install -e .
fi

if pip install ".[pymupdf]" 2>/dev/null; then
  echo "Optional pymupdf installed."
else
  echo "Optional pymupdf not installed (ok; app will use pypdf fallback)."
fi

if ! command -v pdftoppm &>/dev/null; then
  echo "Reminder: install poppler (e.g. brew install poppler or apt-get install poppler-utils)."
fi

echo ""
echo "Activate with: source .venv/bin/activate"
echo "Run: ./pdf_to_md.py --help"

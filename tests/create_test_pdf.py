"""
Utility script to create a simple test PDF for testing.

This creates a minimal PDF with text and a simple table for use in tests.
"""
import os
from typing import List

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_test_pdf(
    output_path: str = "tests/fixtures/test_simple.pdf",
    num_pages: int = 5,
) -> str:
    """
    Create a simple test PDF with multiple pages.

    Each page has distinct content including the page number for verification.

    Args:
        output_path: Path where the PDF should be saved.
        num_pages: Number of pages to create (default: 5).

    Returns:
        The output path where the PDF was saved.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    for page_num in range(1, num_pages + 1):
        # Title with page number
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, f"Test Document - Page {page_num}")

        # Unique content per page
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 100, f"This is page {page_num} of {num_pages}.")
        c.drawString(50, height - 130, f"Unique identifier: PAGE_{page_num}_CONTENT")

        # Simple table with page-specific data
        c.drawString(50, height - 180, "Column A | Column B | Column C")
        c.drawString(50, height - 200, "-" * 40)
        for row in range(3):
            c.drawString(
                50,
                height - 220 - (row * 20),
                f"Row {row + 1}   | P{page_num}R{row + 1}  | Value {page_num}.{row + 1}"
            )

        # Add page break (except for last page)
        if page_num < num_pages:
            c.showPage()

    c.save()
    print(f"Test PDF created at: {output_path} ({num_pages} pages)")
    return output_path


def create_benchmark_fixture_pdf(
    output_path: str,
    num_pages: int = 10,
    title_prefix: str = "Benchmark Document",
) -> str:
    """Create deterministic benchmark fixture PDF.

    Args:
        output_path: Path where the PDF should be written.
        num_pages: Number of pages to include.
        title_prefix: Prefix used in per-page title lines.

    Returns:
        Path to the generated fixture PDF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf_canvas = canvas.Canvas(output_path, pagesize=letter)
    _, page_height = letter

    for page_num in range(1, num_pages + 1):
        pdf_canvas.setFont("Helvetica-Bold", 18)
        pdf_canvas.drawString(
            50,
            page_height - 50,
            f"{title_prefix} - Page {page_num}",
        )

        pdf_canvas.setFont("Helvetica", 11)
        for line_index in range(18):
            y_pos = page_height - 90 - (line_index * 28)
            marker = (
                f"FIXTURE_PAGE_{page_num}_LINE_{line_index}_"
                f"VALUE_{page_num * (line_index + 1)}"
            )
            pdf_canvas.drawString(50, y_pos, marker)

        if page_num < num_pages:
            pdf_canvas.showPage()

    pdf_canvas.save()
    return output_path


def ensure_default_fixture_pdfs() -> List[str]:
    """Generate deterministic fixture PDFs used by tests and benchmarks.

    Returns:
        List of generated fixture PDF paths.
    """
    fixture_dir = os.path.join("tests", "fixtures")
    os.makedirs(fixture_dir, exist_ok=True)
    small_path = os.path.join(fixture_dir, "benchmark_small.pdf")
    if not os.path.exists(small_path):
        create_benchmark_fixture_pdf(
            output_path=small_path,
            num_pages=5,
            title_prefix="Benchmark Small",
        )

    large_path = os.path.join(fixture_dir, "benchmark_large.pdf")
    if not os.path.exists(large_path):
        create_benchmark_fixture_pdf(
            output_path=large_path,
            num_pages=12,
            title_prefix="Benchmark Large",
        )

    fixture_paths = [small_path, large_path]
    return fixture_paths


if __name__ == "__main__":
    os.makedirs("tests/fixtures", exist_ok=True)
    create_test_pdf()
    for fixture in ensure_default_fixture_pdfs():
        print(f"Fixture PDF ready: {fixture}")


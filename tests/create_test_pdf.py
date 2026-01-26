"""
Utility script to create a simple test PDF for testing.

This creates a minimal PDF with text and a simple table for use in tests.
"""
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


if __name__ == "__main__":
    import os

    os.makedirs("tests/fixtures", exist_ok=True)
    create_test_pdf()


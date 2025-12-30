"""
Utility script to create a simple test PDF for testing.

This creates a minimal PDF with text and a simple table for use in tests.
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_test_pdf(output_path: str = "tests/fixtures/test_simple.pdf"):
    """
    Create a simple test PDF with text and a table.

    Args:
        output_path: Path where the PDF should be saved.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Test Document")

    # Paragraph
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "This is a test page for PDF processing.")

    # Simple table
    c.drawString(50, height - 150, "Item 1 | Item 2 | Item 3")
    c.drawString(50, height - 170, "Value A | Value B | Value C")
    c.drawString(50, height - 190, "Data 1  | Data 2  | Data 3")

    # Second page (optional)
    c.showPage()
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Page 2")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "This is the second page of the test document.")

    c.save()
    print(f"Test PDF created at: {output_path}")


if __name__ == "__main__":
    import os

    os.makedirs("tests/fixtures", exist_ok=True)
    create_test_pdf()


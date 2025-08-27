"""
PDF Builder: Reconstruct structured PDF from detected blocks & recognized text.
- Uses reportlab for PDF generation
- Maintains block positions (scaled for page size)
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


def build_pdf(output_path, page_blocks, page_size=A4):
    """
    Build PDF from page text blocks.

    Args:
        output_path (str): Path to save PDF file
        page_blocks (List[Dict]): list of block dicts:
            {
              "bbox": (x, y, w, h),
              "text": "recognized text",
              "block_type": "text" / "table" / etc.
            }
        page_size (tuple): reportlab page size (default A4)
    """
    c = canvas.Canvas(output_path, pagesize=page_size)
    width, height = page_size

    for block in page_blocks:
        x, y, w, h = block["bbox"]
        text = block["text"]

        # Scale coordinates to page size (assuming bbox from image dims)
        # Here we assume input image height ~1000, width ~800 (normalize if needed)
        scale_x = width / 800.0
        scale_y = height / 1000.0

        x *= scale_x
        y *= scale_y
        w *= scale_x
        h *= scale_y

        # In reportlab, y=0 is bottom — need to flip coordinate
        y_pdf = height - y - h

        if block["block_type"] == "text":
            text_obj = c.beginText(x, y_pdf + h)
            text_obj.setFont("Times-Roman", 11)

            for line in text.split("\n"):
                text_obj.textLine(line.strip())
            c.drawText(text_obj)

        elif block["block_type"] == "table":
            # Very basic: just print text block
            c.rect(x, y_pdf, w, h, stroke=1, fill=0)
            c.drawString(x + 5, y_pdf + h - 15, text[:50] + "...")

        else:
            # Default fallback
            c.drawString(x, y_pdf + h, text)

    c.showPage()
    c.save()
    print(f"✅ PDF saved at {output_path}")

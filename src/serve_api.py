"""
Serve API / CLI for full HTR pipeline
- Input: handwritten PDF or image(s)
- Output: structured typed PDF
"""

import argparse
import os

from src.preprocess import pdf_to_images, preprocess_image
from src.layout import detect_blocks
from src.segmentation import segment_lines
from src.htr_model import load_model, recognize_line
from src.postprocess import clean_text
from src.pdf_builder import build_pdf


def process_document(input_path, output_path, model_ckpt="models/htr_model.pth"):
    """
    Run full HTR pipeline on PDF or image and save typed PDF.
    """
    # 1. Convert PDF → images
    if input_path.lower().endswith(".pdf"):
        pages = pdf_to_images(input_path)
    else:
        raise ValueError("Currently only PDF input supported.")

    # 2. Load HTR model
    model = load_model(model_ckpt)

    all_page_blocks = []

    for i, page in enumerate(pages):
        # 3. Preprocess page
        gray, bin_img = preprocess_image(page)

        # 4. Detect blocks
        blocks = detect_blocks(gray)

        page_blocks = []
        for bbox in blocks:
            x, y, w, h = bbox

            # 5. Segment into lines
            lines = segment_lines(gray[y:y+h, x:x+w])

            # 6. Recognize each line
            recognized_text = []
            for line_img in lines:
                text = recognize_line(model, line_img)
                recognized_text.append(clean_text(text))

            full_block_text = "\n".join(recognized_text)

            page_blocks.append({
                "bbox": bbox,
                "text": full_block_text,
                "block_type": "text"
            })

        all_page_blocks.extend(page_blocks)

    # 7. Build PDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    build_pdf(output_path, all_page_blocks)


def main():
    parser = argparse.ArgumentParser(description="HTR Pipeline - Handwriting to Typed PDF")
    parser.add_argument("--input", type=str, required=True, help="Path to input PDF")
    parser.add_argument("--output", type=str, required=True, help="Path to save output typed PDF")
    parser.add_argument("--model", type=str, default="models/htr_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    process_document(args.input, args.output, args.model)


if __name__ == "__main__":
    main()

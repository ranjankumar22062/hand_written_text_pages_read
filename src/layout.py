

import cv2
import numpy as np
from typing import List, Tuple, Dict


def detect_blocks(image: np.ndarray, method: str = "mvp") -> List[Dict]:
    
    if method == "mvp":
        return _detect_blocks_mvp(image)
    else:
        raise NotImplementedError("Advanced layout (Detectron2/PaddleOCR) not integrated yet.")


def _detect_blocks_mvp(image: np.ndarray) -> List[Dict]:
    """
    Simple block detection using morphological operations + contours.
    """
    # Ensure binary image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold (invert so text is white)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to join text lines into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.dilate(threshed, kernel, iterations=2)

    # Find contours of blocks
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 30:  # filter noise
            blocks.append({"bbox": (x, y, w, h), "type": "text"})

    # Sort top-to-bottom, left-to-right
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    return blocks

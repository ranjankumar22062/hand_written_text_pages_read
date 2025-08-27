"""
Line and Word Segmentation for HTR.
"""

import cv2
import numpy as np
from typing import List, Dict


def segment_lines(block_img: np.ndarray) -> List[np.ndarray]:
    """
    Segment a text block into individual lines.
    
    Args:
        block_img (np.ndarray): Cropped block image (binary or grayscale).
    
    Returns:
        List[np.ndarray]: List of line images.
    """
    # Ensure binary
    if len(block_img.shape) == 3:
        gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = block_img

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile
    hist = np.sum(binary, axis=1)
    
    # Detect lines by gaps
    lines = []
    h, w = binary.shape
    start, end = None, None

    for i in range(h):
        if hist[i] > 0 and start is None:
            start = i
        elif hist[i] == 0 and start is not None:
            end = i
            line_img = binary[start:end, :]
            if line_img.shape[0] > 10:  # avoid noise
                lines.append(line_img)
            start, end = None, None

    return lines


def segment_words(line_img: np.ndarray) -> List[np.ndarray]:
    """
    Segment a line into words using vertical projection.
    """
    # Ensure binary
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection profile
    hist = np.sum(binary, axis=0)

    words = []
    start, end = None, None
    w = binary.shape[1]

    for j in range(w):
        if hist[j] > 0 and start is None:
            start = j
        elif hist[j] == 0 and start is not None:
            end = j
            word_img = binary[:, start:end]
            if word_img.shape[1] > 5:  # avoid speckles
                words.append(word_img)
            start, end = None, None

    return words

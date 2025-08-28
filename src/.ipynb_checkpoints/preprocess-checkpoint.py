

import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
  
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    images = [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) for pil in pil_images]
    return images


def preprocess_image(img: np.ndarray):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # 3. Deskew using image moments
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 4. Adaptive thresholding
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 15
    )

    # 5. Normalize grayscale to [0,1]
    gray_norm = cv2.normalize(gray.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    return gray_norm, bin_img


import numpy as np
from typing import List, Dict

# Example charset (must match htr_model.py)
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()-"
IDX2CHAR = {i: c for i, c in enumerate(CHARSET)}
BLANK_IDX = len(CHARSET)  # last index is CTC blank


def ctc_greedy_decoder(preds: np.ndarray) -> str:
    """
    Greedy decoder for CTC outputs.
    
    Args:
        preds (np.ndarray): shape [T, C] after softmax (time steps x classes).
    
    Returns:
        str: Decoded text string.
    """
    best_path = np.argmax(preds, axis=1)

    # Collapse repeats and remove blanks
    decoded = []
    prev = None
    for p in best_path:
        if p != prev and p != BLANK_IDX:
            decoded.append(IDX2CHAR[p])
        prev = p

    return "".join(decoded)


def clean_text(text: str) -> str:
    """
    Basic text cleanup:
    - remove extra spaces
    - fix punctuation spacing
    """
    import re
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.strip()
    return text


def decode_batch(predictions: np.ndarray) -> List[str]:
    """
    Decode batch of model outputs.
    
    Args:
        predictions (np.ndarray): shape [B, T, C]
    
    Returns:
        List[str]: Decoded & cleaned text per sample
    """
    results = []
    for pred in predictions:
        text = ctc_greedy_decoder(pred)
        text = clean_text(text)
        results.append(text)
    return results

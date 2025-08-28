

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

# ---------- CRNN (trainable) ----------

class CRNN(nn.Module):
    """
    Simple CRNN: small CNN feature extractor -> BiLSTM -> linear -> CTC
    Input: (batch, 1, H, W) grayscale images (height fixed)
    Output: (T, batch, num_classes) logits for CTC
    """
    def __init__(self, img_h: int, nc: int, nclass: int, nh: int = 256):
        super().__init__()
        assert img_h % 16 == 0, "img_h must be multiple of 16 for pooling"
        self.cnn = nn.Sequential(
            # conv block 1
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # /2
            # conv block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # /4
            # conv block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),  # /8 height
            # conv block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1), (2,1)),  # /16 height
            # conv block 5 (reduce width channels)
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        # feature height after CNN must be 1 (we assume img_h/16 -> 1)
        self.img_h = img_h
        self.nh = nh
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        conv = self.cnn(x)  # (B, Cf, Hf, Wf)
        b, c, h, w = conv.size()
        assert h == 1, f"Expected conv feature height == 1, got {h}"
        conv = conv.squeeze(2)  # (B, Cf, Wf)
        conv = conv.permute(2, 0, 1)  # (Wf, B, Cf) -> time-major for RNN
        output = self.rnn(conv)  # (Wf, B, nclass)
        return output  # logits (seq_len, batch, nclass)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # seq_len, batch, 2*hidden
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.linear(t_rec)  # (T*b, nOut)
        output = output.view(T, b, -1)
        return output


# ---------- CRNN helper utils ----------

def resize_and_normalize(img: np.ndarray, img_h: int = 32, img_w_max: int = 800) -> torch.FloatTensor:
    """
    Resize image to fixed height img_h while keeping aspect ratio;
    pad width to nearest multiple if needed. Output: (1, img_h, img_w), normalized [0,1]
    """
    # input img assumed uint8 grayscale (H,W)
    h, w = img.shape[:2]
    new_w = max(32, int(w * (img_h / h)))
    if new_w > img_w_max:
        new_w = img_w_max
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_h, new_w)),
        T.ToTensor(),  # [0,1], shape (1,H,W)
    ])
    tensor = transform(img)
    return tensor  # (C=1, H, W)


def ctc_greedy_decode(probs: torch.Tensor, alphabet: List[str], blank: int = 0) -> List[str]:
    """
    Greedy CTC decoder.
    probs: (T, C) or (T, batch, C) - if batch dims present it handles batch 1
    alphabet: list of characters such that index i corresponds to class i (excluding blank index 0)
    blank: index of blank token in model outputs (default 0)
    """
    if probs.dim() == 3:
        probs = probs[:, 0, :]  # take batch 0 if provided

    # choose max at each time step
    max_idxs = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
    # collapse repeats and remove blanks
    prev = None
    out = []
    for idx in max_idxs:
        if idx != prev and idx != blank:
            out.append(idx)
        prev = idx
    # map indexes to chars (alphabet may assume blank at 0)
    # if alphabet provided with blank removed, adapt indexes accordingly
    # Here assume alphabet list corresponds to class indices starting from 1 (0: blank)
    chars = []
    for i in out:
        if i > 0 and (i - 1) < len(alphabet):
            chars.append(alphabet[i - 1])
    return ["".join(chars)]


# ---------- TrOCR wrapper (quick inference) ----------

class TrOCRWrapper:
    def __init__(self, device: Optional[str] = None, model_name: str = "microsoft/trocr-base-handwritten"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except Exception as e:
            raise RuntimeError("transformers not installed or import failed. Install 'transformers' to use TrOCR.") from e

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def recognize(self, pil_image) -> str:
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_length=256)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text


# ---------- HTRModel wrapper ----------

class HTRModel:
    """
    Unified wrapper. Use backend="trocr" for HuggingFace TrOCR (recommended quick start),
    or backend="crnn" for trainable local CRNN.
    """
    def __init__(self, backend: str = "trocr", crnn_params: dict = None, alphabet: str = None):
        self.backend = backend.lower()
        if self.backend == "trocr":
            try:
                self.trocr = TrOCRWrapper()
            except Exception as e:
                raise
        elif self.backend == "crnn":
            # alphabet is a string of characters, blank will be index 0
            if alphabet is None:
                raise ValueError("alphabet must be provided for CRNN backend")
            self.alphabet = list(alphabet)
            img_h = crnn_params.get("img_h", 32)
            nclass = len(self.alphabet) + 1  # +1 for blank
            self.crnn_model = CRNN(img_h=img_h, nc=1, nclass=nclass, nh=crnn_params.get("nh", 256))
            self.device = crnn_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.crnn_model = self.crnn_model.to(self.device)
        else:
            raise ValueError("backend must be 'trocr' or 'crnn'")

    # TrOCR path
    def recognize_line_pil(self, pil_image) -> str:
        if self.backend != "trocr":
            raise RuntimeError("recognize_line_pil only available for trocr backend")
        return self.trocr.recognize(pil_image)

    # CRNN inference path
    @torch.inference_mode()
    def crnn_predict_line(self, line_img: np.ndarray) -> str:
        """
        Accepts single line image (grayscale uint8 numpy). Returns predicted string.
        """
        if self.backend != "crnn":
            raise RuntimeError("crnn_predict_line only available for crnn backend")
        t = resize_and_normalize(line_img)  # (1,H,W)
        t = t.unsqueeze(0).to(self.device)  # (B=1, C=1, H, W)
        logits = self.crnn_model(t)  # (T, B, C)
        probs = F.log_softmax(logits, dim=2).exp()  # convert to probs
        text = ctc_greedy_decode(probs[:, 0, :], self.alphabet, blank=0)[0]
        return text

    # Optional: save / load CRNN state dict
    def save_crnn(self, path: str):
        if self.backend != "crnn": raise RuntimeError
        torch.save(self.crnn_model.state_dict(), path)

    def load_crnn(self, path: str):
        if self.backend != "crnn": raise RuntimeError
        self.crnn_model.load_state_dict(torch.load(path, map_location=self.device))

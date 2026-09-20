import os
import sys

import torch
from PyQt6.QtWidgets import QApplication

from sam3 import build_sam3_image_model
from sam3.model_builder import download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

from Sam3InteractiveWidget import Sam3DesktopApp

# Directory this script lives in, so assets resolve regardless of cwd.
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def build_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bpe_path = os.path.join(APP_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
    print(f"Using BPE vocab: {bpe_path}")
    print(f"Using device: {device}")

    # Hardwired to SAM 3.1. Downloads facebook/sam3.1 from HuggingFace on first
    # run (requires gated access + `hf auth login`), then loads from local cache.
    checkpoint_path = download_ckpt_from_hf(version="sam3.1")
    print(f"Using SAM 3.1 checkpoint: {checkpoint_path}")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    )
    return Sam3Processor(model, device=device)


def main():
    if torch.cuda.is_available():
        # Match the notebook: enable bfloat16 autocast and inference mode.
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    processor = build_processor()
    app = QApplication(sys.argv)
    window = Sam3DesktopApp(processor)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

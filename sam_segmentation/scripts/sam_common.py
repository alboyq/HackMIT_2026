"""Shared helpers for building the SAM 3.1 processor. Used by run_sample.py and export_snapshot.py."""
import torch


def build_processor(checkpoint_version="sam3.1", confidence_threshold=0.5):
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    checkpoint_path = download_ckpt_from_hf(version=checkpoint_version)
    model = build_sam3_image_model(device=device, checkpoint_path=checkpoint_path, load_from_HF=False)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    return processor, device

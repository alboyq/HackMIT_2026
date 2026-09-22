"""M0 smoke test: load SAM 3, segment one sample image with text prompts, save an overlay.

Usage:
    python scripts/run_sample.py --image data/sample.jpg --prompt "water bottle" --prompt "spool" --prompt "bag"
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_overlay(image: Image.Image, detections: list[dict]) -> Image.Image:
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    rng = np.random.default_rng(0)
    colors = [tuple(int(c) for c in rng.integers(60, 255, 3)) + (110,) for _ in detections]

    for det, color in zip(detections, colors):
        mask = det["mask"]
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = color
        draw_layer = Image.alpha_composite(draw_layer, Image.fromarray(mask_rgba, "RGBA"))
        draw = ImageDraw.Draw(draw_layer)
        x0, y0, x1, y1 = det["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=color[:3] + (255,), width=3)
        label = f"{det['id']} {det['label']} {det['score']:.2f}"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(overlay, draw_layer).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/sample.jpg")
    parser.add_argument("--prompt", action="append", dest="prompts",
                         default=None, help="repeatable, e.g. --prompt cup --prompt block")
    parser.add_argument("--out", default="data/overlay.png")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--checkpoint-version", default="sam3.1", choices=["sam3", "sam3.1"])
    args = parser.parse_args()

    prompts = args.prompts or ["water bottle", "spool", "bag", "cardboard package", "tail light"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device: {device}")
    if device == "cuda":
        # bf16 autocast keeps this within the 6 GB VRAM budget on the RTX 3050.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    t0 = time.perf_counter()
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = download_ckpt_from_hf(version=args.checkpoint_version)
    model = build_sam3_image_model(device=device, checkpoint_path=checkpoint_path, load_from_HF=False)
    processor = Sam3Processor(model, device=device)
    t_load = time.perf_counter() - t0
    print(f"[timing] model load ({args.checkpoint_version}): {t_load:.2f}s")

    image = Image.open(args.image).convert("RGB")

    t0 = time.perf_counter()
    state = processor.set_image(image)
    t_set_image = time.perf_counter() - t0
    print(f"[timing] set_image: {t_set_image:.2f}s")

    detections = []
    next_id = 1
    for prompt in prompts:
        t0 = time.perf_counter()
        output = processor.set_text_prompt(state=state, prompt=prompt)
        t_prompt = time.perf_counter() - t0
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        n_raw = len(scores)
        print(f"[timing] prompt '{prompt}': {t_prompt:.2f}s, {n_raw} raw masks")

        for mask, box, score in zip(masks, boxes, scores):
            if float(score) < args.score_thresh:
                continue
            mask_np = mask.squeeze().cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask).squeeze()
            box_np = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
            detections.append({
                "id": next_id,
                "label": prompt,
                "score": float(score),
                "mask": mask_np,
                "bbox": [int(v) for v in box_np[:4]],
            })
            next_id += 1

    print(f"[result] {len(detections)} masks kept (score >= {args.score_thresh}) out of prompts: {prompts}")

    overlay = draw_overlay(image, detections)
    overlay.save(args.out)
    print(f"[result] overlay saved to {args.out}")


if __name__ == "__main__":
    main()

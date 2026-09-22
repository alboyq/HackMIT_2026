"""M1: clean snapshot pipeline (SAM_segmentation_PRD.md F3-F8, section 5/6).

Runs text prompts, filters raw detections (confidence, min area, dedup/overlap,
excluded classes), keeps top-N by score/size/separation, assigns non-harmonic
flicker frequencies, freezes a snapshot_id, and writes:
    snapshots/<snapshot_id>/image.jpg
    snapshots/<snapshot_id>/mask_<id>.png     (one per kept object, 0/255 L-mode)
    snapshots/<snapshot_id>/overlay.png       (debug view, F10)
    snapshots/<snapshot_id>/snapshot.json     (scene.snapshot contract, both PRDs)

Usage:
    python scripts/export_snapshot.py --image data/sample3.jpg \
        --prompt "can" --prompt "bottle" --prompt "bag"
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam_common import build_processor

# Default set from SAM_segmentation_PRD.md section 6. No entry is an integer
# multiple of another. Assigned in this order to the top-ranked kept objects.
FREQ_TABLE = [8.57, 10.0, 12.0, 15.0]

DEFAULT_EXCLUDE_PROMPTS = ["table", "background", "robot arm", "hand"]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def collect_detections(processor, state, prompts, score_thresh):
    detections = []
    for prompt in prompts:
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        for mask, box, score in zip(masks, boxes, scores):
            if float(score) < score_thresh:
                continue
            mask_np = mask.squeeze().cpu().numpy().astype(bool)
            box_np = box.cpu().numpy()
            detections.append({
                "label": prompt,
                "score": float(score),
                "mask": mask_np,
                "bbox": [float(v) for v in box_np[:4]],
            })
    return detections


def filter_and_select(detections, excluded, image_wh, min_area_frac, dedup_iou,
                       separation_frac, max_n):
    img_w, img_h = image_wh
    img_area = img_w * img_h
    diag = (img_w ** 2 + img_h ** 2) ** 0.5

    # Drop small masks.
    kept = [d for d in detections if d["mask"].sum() >= min_area_frac * img_area]

    # Drop anything that overlaps an excluded-class detection (table/background/arm).
    if excluded:
        survivors = []
        for d in kept:
            if any(mask_iou(d["mask"], e["mask"]) > 0.5 for e in excluded):
                continue
            survivors.append(d)
        kept = survivors

    # Dedup: same physical object hit by multiple prompts -> keep the higher score.
    kept.sort(key=lambda d: d["score"], reverse=True)
    deduped = []
    for d in kept:
        if any(mask_iou(d["mask"], k["mask"]) > dedup_iou for k in deduped):
            continue
        deduped.append(d)

    # Top-N by score, enforcing minimum centroid separation.
    def centroid(d):
        x0, y0, x1, y1 = d["bbox"]
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    selected = []
    for d in deduped:
        c = centroid(d)
        too_close = any(
            ((c[0] - cx) ** 2 + (c[1] - cy) ** 2) ** 0.5 < separation_frac * diag
            for cx, cy in (centroid(s) for s in selected)
        )
        if too_close:
            continue
        selected.append(d)
        if len(selected) >= max_n:
            break

    return selected


def next_snapshot_id(snapshots_dir: Path) -> str:
    # ISO 8601 basic format (no colons) so the id is also a valid Windows directory name.
    date_prefix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    existing = sorted(snapshots_dir.glob("*"))
    return f"{date_prefix}_{len(existing) + 1:03d}"


def draw_overlay(image, objects):
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    rng = np.random.default_rng(0)
    for obj in objects:
        color = tuple(int(c) for c in rng.integers(60, 255, 3)) + (110,)
        mask = obj["_mask"]
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask] = color
        draw_layer = Image.alpha_composite(draw_layer, Image.fromarray(mask_rgba, "RGBA"))
        draw = ImageDraw.Draw(draw_layer)
        x0, y0, x1, y1 = obj["bbox_px"]
        draw.rectangle([x0, y0, x1, y1], outline=color[:3] + (255,), width=3)
        label = f"{obj['id']} {obj['label']} {obj['score']:.2f} {obj['frequency_hz']}Hz"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(overlay, draw_layer).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts", required=True)
    parser.add_argument("--exclude-prompt", action="append", dest="exclude_prompts", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-area-frac", type=float, default=0.002,
                         help="drop masks smaller than this fraction of the image area")
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--separation-frac", type=float, default=0.05,
                         help="min centroid distance between kept objects, as a fraction of the image diagonal")
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--out-dir", default="snapshots")
    args = parser.parse_args()

    exclude_prompts = args.exclude_prompts if args.exclude_prompts is not None else DEFAULT_EXCLUDE_PROMPTS

    processor, device = build_processor()
    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size

    t0 = time.perf_counter()
    state = processor.set_image(image)
    print(f"[timing] set_image: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    raw = collect_detections(processor, state, args.prompts, args.score_thresh)
    excluded = collect_detections(processor, state, exclude_prompts, args.score_thresh) if exclude_prompts else []
    print(f"[timing] prompts ({len(args.prompts)} include, {len(exclude_prompts)} exclude): "
          f"{time.perf_counter() - t0:.2f}s, {len(raw)} raw detections")

    kept = filter_and_select(
        raw, excluded, (img_w, img_h),
        args.min_area_frac, args.dedup_iou, args.separation_frac, args.max_n,
    )
    print(f"[result] {len(raw)} raw -> {len(kept)} kept (max {args.max_n})")

    if len(kept) > len(FREQ_TABLE):
        raise SystemExit(f"max-n ({args.max_n}) exceeds FREQ_TABLE size ({len(FREQ_TABLE)})")

    snapshots_dir = Path(args.out_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = next_snapshot_id(snapshots_dir)
    snapshot_dir = snapshots_dir / snapshot_id
    snapshot_dir.mkdir(parents=True)

    objects = []
    for idx, det in enumerate(kept, start=1):
        mask = det["mask"]
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        centroid = [int(xs.mean()), int(ys.mean())]
        area_px = int(mask.sum())
        freq = FREQ_TABLE[idx - 1]

        mask_filename = f"mask_{idx}.png"
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(snapshot_dir / mask_filename)

        objects.append({
            "id": idx,
            "label": det["label"],
            "score": round(det["score"], 4),
            "frequency_hz": freq,
            "bbox_px": [x0, y0, x1, y1],
            "centroid_px": centroid,
            "area_px": area_px,
            "mask_path": f"snapshots/{snapshot_id}/{mask_filename}",
            "mask_url": f"/snapshots/{snapshot_id}/{mask_filename}",
            "_mask": mask,
        })

    image_filename = "image.jpg"
    image.save(snapshot_dir / image_filename, quality=95)

    overlay = draw_overlay(image, objects)
    overlay.save(snapshot_dir / "overlay.png")

    freq_table_out = {str(o["id"]): o["frequency_hz"] for o in objects}
    for o in objects:
        del o["_mask"]

    snapshot_json = {
        "type": "scene.snapshot",
        "snapshot_id": snapshot_id,
        "image_size": [img_w, img_h],
        "image_path": f"snapshots/{snapshot_id}/{image_filename}",
        "objects": objects,
        "frequency_table": freq_table_out,
    }
    with open(snapshot_dir / "snapshot.json", "w") as f:
        json.dump(snapshot_json, f, indent=2)

    print(f"[result] snapshot saved to {snapshot_dir}")
    print(f"[result] objects: {[(o['id'], o['label'], o['frequency_hz']) for o in objects]}")


if __name__ == "__main__":
    main()

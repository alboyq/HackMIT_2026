# SAM3 Interactive Segmentation — Desktop App

A PyQt6 desktop GUI for interactive image segmentation with
[SAM3](https://github.com/facebookresearch/sam3). Load an image (file or URL),
type a text prompt (e.g. `"person"`, `"dog"`), or draw positive/negative box
prompts, and see live segmentation masks with adjustable confidence.

![screenshot](assets/screenshot.png)

## Features

- **Text prompts** — segment by describing what you want.
- **Box prompts** — draw positive (green) or negative (red) boxes to refine.
- **Confidence threshold** slider for filtering detections.
- Load images from **local file** or **URL**.
- Native file dialog on Windows/WSL, HEIC/HEIF support (optional).

## Requirements

This app is hardwired to the **SAM 3.1** checkpoint (`facebook/sam3.1`). The
weights are **not** bundled here — on first run they download from HuggingFace
into your local cache. That repo is **gated**: you must
[request access](https://huggingface.co/facebook/sam3.1) and authenticate with
`hf auth login` once per machine. After the first download it runs from cache
(offline-capable via `HF_HUB_OFFLINE=1`).

Install SAM3 first:

```bash
# 1. Clone and install SAM3 (provides the `sam3` Python package + weights)
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
# Download model checkpoints per the SAM3 repo instructions.
```

Then install this app's own dependencies:

```bash
cd sam3-interactive-app
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the app falls back to CPU automatically.

## Usage

```bash
python app.py
```

- **Open Image…** or paste an image URL and click **Load URL**.
- Type a **text prompt** and click **Segment** (or press Enter).
- Draw a box on the canvas to add a **box prompt**; toggle Positive/Negative.
- Adjust the **Confidence** slider to filter results.
- **Clear All Prompts** to start over.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Entry point: builds the SAM3 processor and launches the GUI. |
| `Sam3InteractiveWidget.py` | The PyQt6 `Sam3DesktopApp` window and all UI/rendering logic. |
| `assets/bpe_simple_vocab_16e6.txt.gz` | BPE vocab used by the SAM3 text encoder. |

## License

The SAM3 model and code are licensed by Meta; see the upstream
[SAM3 repository](https://github.com/facebookresearch/sam3). This app wraps that
library for interactive desktop use.

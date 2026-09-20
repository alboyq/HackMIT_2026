# SAM 3.1 setup — M0 (sample image)

Local machine: RTX 3050 (6 GB VRAM), driver supports up to CUDA 12.7. SAM 3 requires
Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ — a `sam3` conda env (Python 3.12) is being
created for this.

## Steps

1. **Conda env** (in progress): `conda create -n sam3 python=3.12`
2. **PyTorch with CUDA**: use the `cu126` wheel, not `cu128` — the installed driver
   only advertises CUDA 12.7 support, and a `cu128` build needs a driver new enough
   for CUDA 12.8, which this one is not.
   ```bash
   conda activate sam3
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
3. **Clone + install SAM 3**:
   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   ```
4. **Hugging Face access (you do this)**:
   - Go to https://huggingface.co/facebook/sam3, click "Request access", accept the license.
   - Generate an access token: HF Settings → Access Tokens.
   - In your own terminal: `hf auth login` and paste the token when prompted.
   - Tell me once this is done — I won't handle the token myself.
5. **Download checkpoint**:
   ```bash
   hf download facebook/sam3 --local-dir ./checkpoints
   ```
6. **Run the M0 smoke test**:
   ```bash
   python scripts/run_sample.py --image data/sample.jpg
   ```
   This loads the model once, runs `set_image` + a few text prompts against
   `data/sample.jpg` (the HackIITB table photo — filament spool, water bottles,
   cardboard package, tote bag, tail light), prints load/set_image/per-prompt
   timings, and writes `data/overlay.png` with masks + labels + scores drawn on.

## Extra deps needed on Windows (not in the official README)

`pip install -e .` doesn't pull everything actually imported at `import sam3` time
(the package's `__init__.py` eagerly imports training/video/tracker code paths even
for basic image inference). Had to additionally install, in the `sam3` conda env:

```bash
pip install einops                 # sam3/sam/rope.py
pip install pycocotools             # sam3/train/data/coco_json_loaders.py (transitively imported)
pip install psutil                  # sam3/model/sam3_video_predictor.py (transitively imported)
pip install triton-windows          # no official `triton` wheel for Windows; this is the
                                     # community drop-in (installs as `triton`)
pip install "setuptools<81"         # setuptools >=81 dropped pkg_resources, which
                                     # sam3/model_builder.py imports directly
pip install "numpy<2,>=1.26"        # sam3 pins numpy<2; installing the GUI's matplotlib
                                     # afterwards silently bumps numpy to 2.x — reinstall
                                     # numpy<2 last if you install matplotlib/GUI deps
```

## Prompt choice: use "item", not per-object nouns

Specific nouns ("can", "bottle", "case") require guessing every object type in
advance and still miss things — six different phrasings of "AirPods case"
(`earbuds case`, `case`, `remote control`, `pouch`, `black box`, `electronic
device`) all returned zero detections on a real 4-object desk photo. The
single generic prompt **`item`** caught all 4 objects in one shot, including
the AirPods case, all scoring 0.67-0.71 (well above a normal 0.5 threshold):

```bash
python scripts/export_snapshot.py --image data/sample3.jpg --prompt "item" --score-thresh 0.5 --max-n 4
```

`object`/`objects`/`thing` were tried too and gave much weaker signal
(0.03-0.37) — `item` is specifically the concept SAM 3.1 grounds well as a
catch-all. Since the prompt text is no longer a meaningful per-object label
(every object is generically "item"), the dashboard displays objects by ID
("TARGET N") rather than the prompt string.

## Risk: VRAM

SAM 3 is a larger detector+tracker foundation model; 6 GB may not be enough,
especially in fp32. If `run_sample.py` OOMs:
- try `torch.autocast` / half precision if the repo's API supports it, or
- fall back to a cloud/Colab GPU and serve masks over the network (per PRD §10),
  or a lighter segmenter (MobileSAM/FastSAM) as the PRD's fallback option.

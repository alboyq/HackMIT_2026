# Gaze3d Server Configuration (GB10)

## Current Working Setup

**Location**: `~/gaze3d_server/gaze3d` on remote box  
**Python venv**: `~/gaze3d_server/gaze3d/.venv`  
**PyTorch**: `torch==2.14.0+cu130` (CUDA 13.0 required for GB10 compute capability 12.1)

### Start Command (Full 4-Model Ensemble)
```bash
cd ~/gaze3d_server/gaze3d && source .venv/bin/activate && \
nohup python -m gaze3d.server --source push --port 8010 --preload > gaze3d.log 2>&1 &
```

### Available Models
| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `unigaze_h14_joint` | Largest (ViT-H) | Slowest | Most accurate |
| `puregaze_r50` | Medium (ResNet50) | Fast | Good accuracy |
| `gazetr_hybrid` | Medium | Medium | Transformer-based |
| `xgaze_resnet18` | Smallest | Fastest | Decent accuracy |

### Faster Single-Model Configs

**Fast (good enough for demos)**:
```bash
python -m gaze3d.server --source push --port 8010 --models puregaze_r50
```

**Fastest (lower accuracy)**:
```bash
python -m gaze3d.server --source push --port 8010 --models xgaze_resnet18
```

**Balanced (2 models)**:
```bash
python -m gaze3d.server --source push --port 8010 --models puregaze_r50,xgaze_resnet18
```

### Performance Options
- `--fp16` — Half precision, halves VRAM, ~0.05° accuracy cost
- `--no-tta` — Disable test-time augmentation (flip), faster but less stable
- `--preload` — Load models at startup instead of on first frame

### SSH Tunnel (run on Windows)
```powershell
ssh -J jump@129.153.206.6 -L 8010:127.0.0.1:8010 -N asus@10.10.10.4
```

### API Endpoints
- `GET /api/status` — Check server status and loaded models
- `WS /ws` — WebSocket for frame push and gaze data

### Calibration
The gaze3d server stores calibration server-side. After calibration completes via the `/iris` page, the fitted model persists until the server restarts. To save/load calibration profiles:
- `{"cmd": "save", "name": "myprofile"}` → saves to disk
- `{"cmd": "load", "name": "myprofile"}` → restores from disk
- `{"cmd": "clear"}` → clears current calibration

### Measured Performance (GB10, 4 models)
- Full ensemble: ~8 Hz (125ms per frame)
- Single model (puregaze): ~30+ Hz

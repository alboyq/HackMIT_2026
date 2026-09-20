# Handoff PRD — SSVEP Brain–Robot Interface (HackMIT-style demo)

**Last updated**: 2026-09-20 16:20 UTC+8

## 1. What this project is

A "look at an object, robot fetches it" demo with **four alternative selection
methods** sharing one backend and one object-detection pipeline:

1. **SSVEP** (`/stimulus`, `/dashboard`) — objects flicker at distinct frequencies;
   an EEG decoder reads which frequency the wearer's visual cortex is resonating
   with.
2. **Gaze (head pose)** (`/gaze`) — turn your head toward an object, dwell, open
   your jaw to confirm.
3. **Iris tracking** (`/iris`) — precise eye tracking using gaze3d ensemble.
   **Now working locally on Windows with RTX 3050.**
4. **Voice** (`/voice`) — say the object name. Now supports ElevenLabs TTS.

All four end in the same spoken confirmation and (eventually) the same robot
pipeline.

## 2. Repo layout

```
C:\Users\User\Downloads\IITB\
  sam_segmentation\          Python, SAM 3.1 object segmentation (done, working)
    scripts\export_snapshot.py   the pipeline entry point
    snapshots\                    output: snapshot.json + per-object mask PNGs + image.jpg
  flicker_dashboard\
    backend\                  FastAPI hub, port 8000
      main.py                 WebSocket relay + REST endpoints
      cloud_ai.py             vision-naming + TTS provider logic
      .env                    API keys (GEMINI_API_KEY, ELEVENLABS_API_KEY)
    frontend\                 Vite + React + TS + Tailwind v4 + framer-motion, port 5173
      src/pages/{StimulusPage,DashboardPage,GazePage,VoicePage,IrisPage}.tsx
      src/features/gaze/*      head-pose gaze (MediaPipe FaceLandmarker, working)
      src/features/iris/*      iris tracking (gaze3d integration, working locally)
  gaze3d_src\gaze3d\          Local gaze3d server (runs on Windows RTX 3050)
    gaze3d\pipeline.py        Main pipeline with auto-load calibration
    gaze3d\server.py          FastAPI server, port 8012
    gaze3d\filters.py         OneEuro filter for gaze smoothing
    profiles\                 Calibration profiles (default.json auto-loads)
  calibration_goldmine\       Permanent backup of calibration data
    BEST_calibration.json     Best calibration profile
    session_*.jsonl           Raw calibration samples
  HANDOFF_PRD.md              this file
```

## 3. Running the system

### Start all services (Windows):
```powershell
# Backend (port 8000)
cd flicker_dashboard\backend
.\.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (port 5173)
cd flicker_dashboard\frontend
npm run dev

# Gaze3d server (port 8012) - LOCAL on Windows
cd gaze3d_src\gaze3d
$env:PYTHONUTF8="1"
.\.venv\Scripts\activate
python -m gaze3d.server --source push --port 8012
```

### API Keys (.env file in backend/):
```
GEMINI_API_KEY=your_key_here      # https://aistudio.google.com/app/apikey
ELEVENLABS_API_KEY=your_key_here  # https://elevenlabs.io/app/settings/api-keys
```

## 4. Iris tracking (`/iris`) — NOW WORKING

### Architecture
- **Local inference**: gaze3d runs on Windows laptop RTX 3050 (port 8012)
- **Model**: `puregaze_r50` (single lightweight model for speed)
- **Pipeline**: Webcam → JPEG push → MediaPipe face → PnP head pose → gaze model → calibration → screen point

### Key features implemented:
1. **Auto-load calibration**: Server loads `profiles/default.json` on startup
2. **29-point calibration**: 5×5 grid + 4 extra corners, 10 seconds per point
3. **Edge gain (1.4×)**: Amplifies gaze at screen borders to fix underestimation
4. **15-second dwell to lock**: Stare at object for 15s to select (no jaw gesture)
5. **Calibration backup**: All calibration data saved in `calibration_goldmine/`

### Frontend settings (`useRemoteIrisGaze.ts`):
```typescript
const GAZE3D_WS_URL = `ws://${window.location.hostname}:8012/ws`  // local server
const PUSH_INTERVAL_MS = 30   // ~33 FPS
const EDGE_GAIN = 1.4         // 40% boost at screen edges
const JPEG_QUALITY = 0.5
```

### Calibration settings (`IrisCalibration.tsx`):
```typescript
const DWELL_MS = 10000   // 10 seconds per static point
const MOTION_MS = 10000  // 10 seconds per motion trial
// Total: 29 static + 5 motion = 34 points × ~10s = ~6 minutes
```

### Server-side filter (`filters.py`):
```python
min_cutoff = 2.0   # Higher = less smoothing
beta = 0.08        # Higher = faster response to velocity
```

### Calibration protocol:
1. 5×5 grid (25 points) covering full screen
2. 4 extra corner points for edge accuracy
3. 5 motion trials (yaw, pitch, depth, lateral, roll)
4. User should move head while keeping eyes on target
5. Calibration auto-saves as `default.json`

## 5. Vision + TTS — Cloud APIs integrated

### Priority order:
**Vision**: Gemini → Local Ollama → OpenAI → Groq → OpenRouter
**TTS**: ElevenLabs → Local Kokoro → OpenAI → Browser

### Gemini Vision (free tier):
- 15 requests/minute, 1500/day
- Fast, good quality
- Set `GEMINI_API_KEY` in `.env`

### ElevenLabs TTS (free tier):
- ~10,000 characters/month
- High quality voice synthesis
- Set `ELEVENLABS_API_KEY` in `.env`
- Default voice: Rachel (21m00Tcm4TlvDq8ikWAM)

### Frontend TTS (`speech.ts`):
- Automatically tries backend TTS (ElevenLabs) first
- Falls back to browser `speechSynthesis` if unavailable
- Uses Web Audio API for playback

### Voice recognition improvements:
- Filters short interim results (< 3 chars) to reduce noise
- 300ms restart delay prevents rapid cycling
- Silently ignores network errors (auto-retries)
- Reduced speech-busy delay for faster response

## 6. Remote compute (optional, for heavy models)

If running full 4-model ensemble instead of local single-model:
```bash
ssh -J jump@129.153.206.6 asus@10.10.10.4
```

SSH tunnel for remote gaze3d:
```bash
ssh -N -L 8010:localhost:8010 -J jump@129.153.206.6 asus@10.10.10.4
```

Then change frontend to use port 8010 instead of 8012.

## 7. SAM Segmentation — DONE

```bash
cd sam_segmentation
conda activate sam3
python scripts/export_snapshot.py --image <path> --prompt "item" --score-thresh 0.5 --max-n 4
```

**Critical**: Use prompt `"item"` (not specific nouns) to catch all objects.

## 8. Calibration data backup

All calibration is backed up in `calibration_goldmine/`:
- `BEST_calibration.json` — working calibration profile
- `session_*.jsonl` — raw sample data (2000+ samples)

To restore calibration:
```powershell
Copy-Item calibration_goldmine\BEST_calibration.json gaze3d_src\gaze3d\profiles\default.json
```

## 9. Known limitations

1. **Single model**: Only `puregaze_r50` is used (not full 4-model ensemble)
   - Accuracy is good but not the 1.3° measured with full ensemble
   - Can run full ensemble on remote GPU if needed

2. **Calibration required**: Each user needs to calibrate (~6 minutes)
   - Calibration saves automatically
   - Persists across restarts via `default.json`

3. **Edge accuracy**: Even with 1.4× edge gain, corners may need more head movement

## 10. Quick troubleshooting

**Gaze3d not responding**:
```powershell
# Check if running
Invoke-WebRequest -Uri "http://localhost:8012/api/status" -UseBasicParsing
# Restart
taskkill /F /IM python.exe
# Start again with commands from section 3
```

**Calibration not loading**:
- Check `gaze3d_src\gaze3d\profiles\default.json` exists
- Server auto-loads on client connect (not on startup)
- Check server console for `[gaze3d] Auto-loaded calibration`

**Images not showing**:
- Make sure backend is running on port 8000
- Check `sam_segmentation/snapshots/` has valid snapshot

**Voice not working**:
- Check `.env` has `ELEVENLABS_API_KEY`
- Falls back to browser speech if no API key
- Check browser console for errors

## 11. GitHub repo

Code is at: https://github.com/alboyq/HackMIT_2026

Relevant folders:
- `gaze3d/` — gaze tracking (also in `gaze3d_src/gaze3d/` locally)
- `flicker_dashboard/` — main application
- `sam_segmentation/` — object segmentation

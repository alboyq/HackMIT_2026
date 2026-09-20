# Flicker + Dashboard

Implements Flicker_and_Dashboard_PRD.md. Two pieces sharing one backend hub:
`/stimulus` (SSVEP flicker for the EEG wearer) and `/dashboard` (mission
control for judges/audience).

## Layout

- `backend/` — FastAPI + WebSocket hub (`main.py`). Relays the section 7
  message contract, serves SAM snapshot files at `/snapshots/...`, and mocks
  every message type not yet produced by a real system (decoder, robot, EEG).
- `frontend/` — Vite + React + TypeScript + Tailwind. Routes: `/stimulus`,
  `/dashboard`.

## Run locally

```bash
# backend (from flicker_dashboard/backend, needs fastapi + uvicorn — already
# in the base Python env used here)
python -m uvicorn main:app --port 8000

# frontend (from flicker_dashboard/frontend)
npm run dev
```

Or use the `flicker-dashboard-frontend` launch config (`IITB/.claude/launch.json`).

## Status

**Done:**
- M0 test-pattern stimulus (`/stimulus`): fixed flicker box, presentation-time
  sine luminance (phase survives dropped frames), self-calibrating measured
  refresh readout, dropped-frame counter, Esc kill switch, seizure warning
  gate. Verified rendering + Esc in-browser; **still needs phone slow-mo or
  photodiode verification on the real target monitor** per the PRD's own
  acceptance criterion — an in-browser readout is not sufficient proof.
- Backend hub: WebSocket relay, static snapshot serving, mock loop for
  decoder/robot/eeg messages, real `scene.snapshot` read from the SAM
  pipeline's latest `snapshots/*/snapshot.json`.
- Dashboard skeleton (`/dashboard`): connection status, mode label, D1 scene
  view (real snapshot + live boxes), D5 decoder confidence bars, D6 pipeline
  stepper (mocked stage), D10 event log. Confirmed working end-to-end against
  the mock server with a real SAM snapshot.

**Not built yet** — pending direction on the PRD's open questions (section 12):
D2 EEG traces, D3 head map, D4 SSVEP spectrum, D7 robot twin, D8 rover map,
D9 ErrP indicator, D11 metrics strip, D12 operator controls / e-stop. Also:
S1/S4/S5 (full multi-object stimulus with select/highlight/rest region — only
the single-box test pattern exists so far), Live/Replay/Scripted mode
switching (mode is hardcoded to "MOCK"), WebGL upgrade for the real
multi-mask stimulus render.

## Bridging from the SAM module

`sam_segmentation/scripts/export_snapshot.py` writes
`sam_segmentation/snapshots/<id>/snapshot.json` + per-object masks + overlay,
matching this project's `scene.snapshot` contract exactly (see
`sam_segmentation/scripts/export_snapshot.py`). The backend hub picks up the
latest one automatically — no manual copying between the two projects.

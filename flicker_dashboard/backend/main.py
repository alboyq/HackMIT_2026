"""Backend hub (Architecture, PRD section 4).

Relays JSON messages (section 7 contract) between modules and the browser over
one WebSocket, and serves snapshot files (image + masks) produced by the SAM
pipeline (../sam_segmentation/snapshots/).

Real today: scene.snapshot (read from the latest SAM snapshot on disk).
Mocked today (until Aditya's decoder / the robot / the EEG headset are wired
in): stim.started/stopped, decoder.scores, decoder.selected, robot.state,
eeg.chunk. Mock mode is what lets the dashboard/stimulus UI be built and
tested now; real producers replace mock_loop's message generation later
without touching the WebSocket contract.
"""
import asyncio
import json
import math
import random
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cloud_ai import name_objects, speak_mp3, tts_provider, vision_provider

ROOT = Path(__file__).resolve().parent.parent.parent
SAM_SNAPSHOTS_DIR = ROOT / "sam_segmentation" / "snapshots"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if SAM_SNAPSHOTS_DIR.exists():
    app.mount("/snapshots", StaticFiles(directory=str(SAM_SNAPSHOTS_DIR)), name="snapshots")


class ConnectionHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = ConnectionHub()


def latest_snapshot() -> dict | None:
    if not SAM_SNAPSHOTS_DIR.exists():
        return None
    candidates = sorted(SAM_SNAPSHOTS_DIR.glob("*/snapshot.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


@app.get("/api/snapshot/latest")
async def get_latest_snapshot():
    snap = latest_snapshot()
    return snap or {"type": "scene.snapshot", "objects": []}


class VisionNameBody(BaseModel):
    images: list[str] = Field(default_factory=list, description="data:image/jpeg;base64,... or raw base64")


def _decode_jpeg(data_url: str) -> bytes:
    raw = data_url.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return __import__("base64").b64decode(raw)


@app.get("/api/ai/status")
async def ai_status():
    return {"vision": vision_provider(), "tts": tts_provider()}


@app.post("/api/vision/name")
async def vision_name(body: VisionNameBody):
    if not body.images:
        return {"names": [], "provider": None}
    try:
        jpegs = [_decode_jpeg(x) for x in body.images]
        names, provider = await asyncio.to_thread(name_objects, jpegs)
        return {"names": names, "provider": provider}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class SpeakBody(BaseModel):
    text: str


@app.post("/api/voice/speak")
async def voice_speak(body: SpeakBody):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty")
    if not tts_provider():
        raise HTTPException(status_code=503, detail="no tts key")
    try:
        audio, content_type = await asyncio.to_thread(speak_mp3, text)
        return Response(content=audio, media_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        snap = latest_snapshot()
        if snap:
            await ws.send_json(snap)
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # robot.command (e.g. e-stop / abort) and any client-originated
            # message are relayed to every other connected client.
            await hub.broadcast(msg)
    except WebSocketDisconnect:
        hub.disconnect(ws)


async def mock_loop():
    """Emits synthetic versions of every non-SAM message in the section 7 contract."""
    t_start_ms = int(time.time() * 1000)
    snap = latest_snapshot()
    ids = [str(o["id"]) for o in snap["objects"]] if snap else ["1", "2", "3"]
    freq_table = (
        {str(o["id"]): o["frequency_hz"] for o in snap["objects"]}
        if snap else {"1": 8.57, "2": 10.0, "3": 12.0}
    )

    await hub.broadcast({
        "type": "stim.started", "snapshot_id": snap["snapshot_id"] if snap else "mock",
        "t_ms": t_start_ms, "measured_refresh_hz": 60.0, "table": freq_table,
    })

    channels = ["Fz", "FCz", "Cz", "CPz", "POz", "Oz", "O1", "O2"]
    t = 0.0
    winner = random.choice(ids)
    tick = 0
    # 0.1s ticks. Pipeline must actually move or the dashboard looks frozen.
    stage_plan = [
        ("SNAPSHOT", 12),
        ("FLICKER", 40),
        ("SELECT", 12),
        ("PLAN", 15),
        ("EXECUTE", 30),
        ("DONE", 20),
    ]
    stage_idx = 0
    stage_left = stage_plan[0][1]
    selected_emitted = False

    while True:
        await asyncio.sleep(0.1)
        tick += 1
        t += 0.1
        stage, _ = stage_plan[stage_idx]

        eeg_data = [
            [math.sin(2 * math.pi * 10 * t + i) + random.gauss(0, 0.2) for _ in range(12)]
            for i in range(len(channels))
        ]
        await hub.broadcast({
            "type": "eeg.chunk", "fs": 125, "channels": channels, "data": eeg_data,
        })

        climb = min(0.92, 0.35 + (stage_plan[stage_idx][1] - stage_left) * 0.012)
        scores = {i: round(random.uniform(0.04, 0.22), 2) for i in ids}
        if stage in ("FLICKER", "SELECT", "PLAN", "EXECUTE", "DONE"):
            scores[winner] = round(climb if stage == "FLICKER" else random.uniform(0.82, 0.96), 2)
        await hub.broadcast({
            "type": "decoder.scores", "window_s": 3.0, "scores": scores, "state": "selecting" if stage == "FLICKER" else stage.lower(),
        })

        await hub.broadcast({
            "type": "robot.state", "stage": stage,
            "joints": {"shoulder_pan": math.sin(t) * 0.2}, "gripper": 1.0 if stage in ("EXECUTE", "DONE") else 0.0,
        })

        if stage == "SELECT" and not selected_emitted:
            selected_emitted = True
            await hub.broadcast({
                "type": "decoder.selected", "id": int(winner), "confidence": scores.get(winner, 0.9),
            })

        stage_left -= 1
        if stage_left <= 0:
            stage_idx = (stage_idx + 1) % len(stage_plan)
            stage_left = stage_plan[stage_idx][1]
            if stage_plan[stage_idx][0] == "SNAPSHOT":
                winner = random.choice(ids)
                selected_emitted = False


@app.on_event("startup")
async def start_mock():
    print(f"[ai] vision={vision_provider() or 'UNCONFIGURED'} tts={tts_provider() or 'browser'}")
    asyncio.create_task(mock_loop())

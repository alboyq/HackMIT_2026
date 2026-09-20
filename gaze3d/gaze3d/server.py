"""FastAPI server: static demo page + WebSocket gaze stream + MJPEG preview.

    python -m gaze3d.server [--port 8010] [--models a,b,c] [--no-tta]
"""
from __future__ import annotations
import argparse, asyncio, json, os, time, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .pipeline import Pipeline, DEFAULT_MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI()
pipe: Pipeline | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "gaze3d.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return JSONResponse(pipe.snapshot())


@app.get("/api/samples")
def samples():
    return JSONResponse(pipe.samples)


async def _mjpeg(getter):
    boundary = b"--frame\r\n"
    last = None
    while True:
        buf = getter()
        if buf is not None and buf is not last:
            last = buf
            yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n"
        await asyncio.sleep(0.08)


@app.get("/preview.mjpg")
def preview():
    return StreamingResponse(_mjpeg(lambda: pipe.preview_jpeg), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/crop.mjpg")
def crop():
    return StreamingResponse(_mjpeg(lambda: pipe.crop_jpeg), media_type="multipart/x-mixed-replace; boundary=frame")


def handle(cmd: dict) -> dict:
    c = cmd.get("cmd")
    if c == "start":
        # width/height only used in --source push (see Pipeline.start's docstring);
        # a --source camera server ignores them and keeps its CLI-configured resolution.
        pipe.start(width=cmd.get("width"), height=cmd.get("height"))
        return dict(ok=True)
    if c == "stop":
        pipe.stop()
        return dict(ok=True)
    if c == "geometry":
        pipe.set_geometry(cmd["width_px"], cmd["height_px"], cmd["pitch_mm"], cmd.get("cam_offset_px", (0.0, 18.0)))
        return dict(ok=True)
    if c == "focal":
        f = pipe.set_focal_from_distance(float(cmd["distance_mm"]))
        return dict(ok=True, focal=f)
    if c == "arm":
        pipe.arm(cmd["point"], cmd["target"], cmd.get("kind", "static"))
        return dict(ok=True)
    if c == "disarm":
        return dict(ok=True, n=pipe.disarm())
    if c == "clear":
        pipe.clear()
        return dict(ok=True)
    if c == "fit":
        return dict(ok=True, report=pipe.fit())
    if c == "recover":
        return dict(ok=True, report=pipe.recover(cmd.get("path")))
    if c == "disk":
        return dict(ok=True, disk=pipe.disk_samples())
    if c == "save":
        return dict(ok=True, path=pipe.save_profile(cmd.get("name", "default")))
    if c == "load":
        return dict(ok=True, report=pipe.load_profile(cmd.get("name", "default")))
    if c == "reset_filter":
        pipe.smoother.reset()
        return dict(ok=True)
    if c == "filter":
        pipe.smoother.f.min_cutoff = float(cmd.get("min_cutoff", pipe.smoother.f.min_cutoff))
        pipe.smoother.f.beta = float(cmd.get("beta", pipe.smoother.f.beta))
        return dict(ok=True)
    return dict(ok=False, error=f"unknown cmd {c}")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_event_loop()

    async def pump():
        last_seq = None
        while True:
            st = pipe.snapshot()
            if st.get("seq") != last_seq or st.get("status") != getattr(pump, "_status", None):
                last_seq = st.get("seq")
                pump._status = st.get("status")
                st["type"] = "frame"
                st["server_t"] = time.time()
                await sock.send_text(json.dumps(st))
            await asyncio.sleep(0.004)

    task = asyncio.create_task(pump())
    try:
        while True:
            # A frame pushed from a browser (see gaze3d/pushcam.py) arrives as a
            # raw binary WS message; every other message is the existing JSON
            # command protocol. Both share this one receive loop so a client can
            # freely interleave "arm"/"fit"/etc. with a steady stream of frames.
            raw = await sock.receive()
            if raw["type"] == "websocket.disconnect":
                break
            data = raw.get("bytes")
            if data is not None:
                # Decoded inline, NOT via run_in_executor like commands below:
                # that executor call is *awaited* right here in this same loop
                # before the next receive() runs, so a slow/blocking handler
                # would stall this connection's ability to read the very next
                # message - including the next pushed frame. A JPEG decode is a
                # few ms, cheap enough to do directly without that risk. (This
                # is also why PushCamera.start() itself never blocks - see its
                # docstring.)
                try:
                    pipe.push_frame(data)
                except Exception:
                    traceback.print_exc()
                continue
            text = raw.get("text")
            if text is None:
                continue
            msg = json.loads(text)
            try:
                # start/fit/load can take seconds: keep the event loop responsive
                res = await loop.run_in_executor(None, handle, msg)
            except Exception as e:
                traceback.print_exc()
                res = dict(ok=False, error=repr(e))
            res.update(type="reply", cmd=msg.get("cmd"), id=msg.get("id"))
            await sock.send_text(json.dumps(res))
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()


app.mount("/", StaticFiles(directory=ROOT), name="static")


def main():
    global pipe
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--source", choices=["camera", "push"], default="camera",
                    help="frame source: a local cv2 device (default), or frames pushed over "
                         "the /ws WebSocket by a remote client (e.g. a browser with no local "
                         "camera on this machine) - see gaze3d/pushcam.py")
    ap.add_argument("--hfov", type=float, default=66.0, help="camera horizontal FOV prior; calibration refines the scale")
    ap.add_argument("--preload", action="store_true", help="load models before serving")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision on CUDA/MPS; ~0.05 deg cost, halves VRAM (use on <=6 GB cards)")
    a = ap.parse_args()
    pipe = Pipeline(models=a.models.split(","), camera_index=a.camera, width=a.width, height=a.height,
                    hfov_deg=a.hfov, tta_flip=not a.no_tta, fp16=a.fp16, source=a.source)
    if a.preload:
        pipe.load()
    print(f"gaze3d → http://localhost:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

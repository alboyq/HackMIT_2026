"""Live view of what the feeding pipeline would SEE on the real wrist camera. Read-only: no arm motion.
  python live_perception.py --food strawberry            (window on the local display + http://localhost:8091)
Shows the named food (box, angular width, grasp class) and the face (mouth, angular width, and what the feed
would be doing at this distance: APPROACH / CRAWL / STOP / HOLD). Type a new food name + Enter in the terminal
to change what it looks for."""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import socket

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_model import CameraModel  # noqa: E402
from face_sensor import FaceSensor  # noqa: E402
from object_sensor import ObjectSensor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--food", default="apple")
ap.add_argument("--device", default="/dev/video0")
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
ap.add_argument("--save", default="", help="write annotated frames here")
args = ap.parse_args()

cal = Path(__file__).resolve().parents[1] / "rl" / "configs" / "feed_calibration.json"
touch = json.loads(cal.read_text())["contact_size_deg_median"] if cal.exists() else 50.9
stop = touch - 4.1                                          # the sim's 1.5 cm stand-off in degrees
_lock, _jpg, food = threading.Lock(), None, {"name": args.food, "dirty": True}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        b = _jpg
                    if b:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body>")


def stdin_loop():
    for line in sys.stdin:
        if line.strip():
            food["name"], food["dirty"] = line.strip(), True


cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, f = cap.read()
if not ok:
    raise SystemExit(f"cannot read {args.device}")
cam = CameraModel(f.shape[1], f.shape[0])
face, obj = FaceSensor(cam), ObjectSensor(cam)
threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever, daemon=True).start()
threading.Thread(target=stdin_loop, daemon=True).start()
show = bool(os.environ.get("DISPLAY"))
print(f"[live] lens {'CALIBRATED' if cam.calibrated else 'UNCALIBRATED (ideal 150-deg fisheye)'}; "
      f"touch {touch:.1f} deg, stop {stop:.1f} deg; stream http://localhost:{args.port}", flush=True)
n = 0
_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       # run_pick.py listens here for what the wrist camera sees
while True:
    ok, f = cap.read()
    if not ok:
        continue
    if food["dirty"]:
        obj.set_food(food["name"])
        food["dirty"] = False
    orr = obj.read(f)
    fr = FaceSensor.confirm(face.read(f), orr.people)
    _udp.sendto(json.dumps({"t": time.time(), "food": food["name"], "seen": bool(orr.seen), "box": orr.box, "w": f.shape[1], "h": f.shape[0]}).encode(), ("127.0.0.1", 8092))
    vis = ObjectSensor.draw(FaceSensor.draw(f.copy(), fr, stop_deg=stop), orr)
    if not cam.calibrated:
        cv2.putText(vis, "LENS UNCALIBRATED - angles approximate", (12, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _lock:
        _jpg = buf.tobytes()
    n += 1
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(args.save) / f"frame{n:03d}.jpg"), vis)
    if n % 10 == 1:
        print(f"[live] food '{food['name']}': " + (f"conf {orr.conf:.2f} {orr.width_deg:.1f} deg class {orr.slot}" if orr.seen else "not found")
              + " | face: " + (f"{fr.width_deg:.1f} deg facing={fr.facing}" if fr.seen else "none"), flush=True)
    if show:
        cv2.imshow("live perception (real wrist camera)", vis)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    if args.frames and n >= args.frames:
        break

"""Live view of the REAL wrist camera, for choosing and checking a lens. Read-only: opens the USB
camera, never touches the arm. Window on the local display + MJPEG on http://localhost:8090.

Overlay: a centre crosshair and a perfectly straight grid. Point the camera at something with straight
edges (door frame, table edge) near the image border: the more the real edge bows away from the grid
line, the more distortion that lens has.    [q] quit   [g] grid on/off   [s] save a frame
"""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

DEV = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
W, H = int(os.environ.get("CAM_W", "1280")), int(os.environ.get("CAM_H", "720"))
PORT = int(os.environ.get("CAM_PORT", "8090"))
_lock, _jpg = threading.Lock(), None


class Handler(BaseHTTPRequestHandler):
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
                        buf = _jpg
                    if buf:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


def main():
    global _jpg
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        raise SystemExit(f"could not open {DEV}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] {DEV} open at {w}x{h}; stream on http://localhost:{PORT}", flush=True)
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever, daemon=True).start()

    show = bool(os.environ.get("DISPLAY"))
    if show:
        cv2.namedWindow("wrist camera (real)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("wrist camera (real)", 1120, 630)
    grid, t0, n, fps, saved = True, time.time(), 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[cam] frame grab failed; retrying", flush=True)
            time.sleep(0.5)
            continue
        n += 1
        if time.time() - t0 >= 1.0:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()
        view = frame.copy()
        if grid:
            for k in range(1, 8):
                cv2.line(view, (w * k // 8, 0), (w * k // 8, h), (0, 255, 0), 1)
                cv2.line(view, (0, h * k // 8), (w, h * k // 8), (0, 255, 0), 1)
            cv2.drawMarker(view, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        for k, text in enumerate([f"{w}x{h}  {fps:4.1f} fps   [q]uit [g]rid [s]ave",
                                  "FOV = 2*atan(W / 2d): W = width seen on a wall at distance d"]):
            cv2.putText(view, text, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _jpg = buf.tobytes()
        if show:
            cv2.imshow("wrist camera (real)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = not grid
            if key == ord("s"):
                saved += 1
                cv2.imwrite(f"/tmp/wrist_frame_{saved}.jpg", frame)
                print(f"[cam] saved /tmp/wrist_frame_{saved}.jpg", flush=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

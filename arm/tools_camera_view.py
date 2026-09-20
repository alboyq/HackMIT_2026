"""Show the first available USB camera on the GX10 monitor, and keep a snapshot on disk.

    DISPLAY=:1 .venv-arm/bin/python arm/tools_camera_view.py

Waits for a camera instead of exiting when none is present, so you can watch the window while
swapping cables. Writes the newest frame to /tmp/cam_latest.jpg every second so it can be
inspected from an SSH session. q or ESC quits, s forces a snapshot.
"""
from __future__ import annotations

import glob
import time

import cv2
import numpy as np

WINDOW = "GX10 camera"
SNAPSHOT = "/tmp/cam_latest.jpg"


def status_frame(lines, w=960, h=540):
    img = np.full((h, w, 3), 24, np.uint8)
    for i, (text, scale, color) in enumerate(lines):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(img, text, ((w - size[0]) // 2, 180 + i * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img


def open_first_camera():
    for path in sorted(glob.glob("/dev/video*")):
        index = int(path.replace("/dev/video", ""))
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            return cap, path
        cap.release()
    return None, None


def main() -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)
    cap = None
    path = None
    last_save = 0.0
    frames = 0
    t0 = time.time()
    dots = 0

    while True:
        if cap is None:
            cap, path = open_first_camera()
            if cap is None:
                dots = (dots + 1) % 4
                frame = status_frame([
                    ("No camera detected" + "." * dots, 1.2, (80, 80, 240)),
                    ("/dev/video* is empty - check the USB cable", 0.7, (200, 200, 200)),
                    ("a charge-only cable powers the camera but carries no data", 0.6, (150, 150, 150)),
                    ("waiting - plug one in and it will appear here", 0.6, (150, 200, 150)),
                ])
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(500) in (ord("q"), 27):
                    break
                continue
            print(f"[camera] opened {path}", flush=True)
            frames, t0 = 0, time.time()

        ok, frame = cap.read()
        if not ok:
            print("[camera] read failed; releasing and rescanning", flush=True)
            cap.release()
            cap = None
            continue

        frames += 1
        fps = frames / max(1e-6, time.time() - t0)
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{path}  {w}x{h}  {fps:.1f} fps", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)

        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        if now - last_save >= 1.0 or key == ord("s"):
            cv2.imwrite(SNAPSHOT, frame)
            last_save = now
        if key in (ord("q"), 27):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

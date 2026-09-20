"""Validate the real-camera sensors WITHOUT the arm: run them on live wrist-camera frames and on still
images, save annotated copies. usage: test_perception.py <out_dir> <food,food,...> [image ...]
With no images given it grabs frames from /dev/video0."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_model import CameraModel  # noqa: E402
from face_sensor import FaceSensor  # noqa: E402
from object_sensor import ObjectSensor  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
foods = sys.argv[2].split(",")
images = sys.argv[3:]

frames = []
if images:
    frames = [(Path(p).stem, cv2.imread(p)) for p in images]
else:
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(15):                                    # let exposure settle
        cap.read()
    for k in range(3):
        ok, f = cap.read()
        if ok:
            frames.append((f"live{k}", f))
        time.sleep(0.4)
    cap.release()
print(f"{len(frames)} frames")

h, w = frames[0][1].shape[:2]
cam = CameraModel(w, h)
print("camera model:", "CALIBRATED" if cam.calibrated else "UNCALIBRATED (ideal 150-deg equidistant fisheye)")
face = FaceSensor(cam)
obj = ObjectSensor(cam)
for name, f in frames:
    if f.shape[1] != w or f.shape[0] != h:
        cam = CameraModel(f.shape[1], f.shape[0])
        face.cam = obj.cam = cam
        h, w = f.shape[:2]
    t0 = time.time()
    fr = face.read(f)
    t_face = time.time() - t0
    obj.set_food(foods[0])
    raw_seen = fr.seen
    fr = FaceSensor.confirm(fr, obj.read(f).people)
    vis = FaceSensor.draw(f.copy(), fr)
    line = f"{name}: face " + (f"{fr.width_deg:.1f} deg (raw width {fr.raw_width_deg:.1f}, height {fr.height_deg:.1f}, facing={fr.facing}, asym {fr.asymmetry:.2f}), bearing "
                               f"({fr.bearing[0]:+.2f},{fr.bearing[1]:+.2f}), mouth_open {fr.mouth_open:.2f}" if fr.seen else ("REJECTED (landmarks fired, no person agrees)" if raw_seen else "NOT FOUND"))
    for food in foods:
        obj.set_food(food)
        t0 = time.time()
        r = obj.read(f)
        vis = ObjectSensor.draw(vis, r)
        line += f" | {food}: " + (f"conf {r.conf:.2f}, {r.width_deg:.1f} deg, class {r.slot}" if r.seen else "not found")
        line += f" [{1000*(time.time()-t0):.0f} ms]"
    print(line + f"  (face {1000*t_face:.0f} ms)")
    cv2.imwrite(str(out / f"{name}.jpg"), vis)

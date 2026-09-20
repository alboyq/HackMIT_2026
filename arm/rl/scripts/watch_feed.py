"""Live view of the hybrid FEED on the GX10 monitor: a side-on view with the arm and the person's head
both in frame, and the WRIST CAMERA view inset (what the real camera will be looking at once the arm has
cocked back level). Real time. CALIBRATE=1 shows the contact-detection runs."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YAM_CAM_RES", "320")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_feed as hf  # noqa: E402

WINDOW, PERIOD = "OpenYAM - feeding", 1.0 / 30.0
NAMES = {"pinch": "2/5 PLACE + PINCH (learned)", "lift": "3/5 LIFT STRAIGHT UP (scripted)",
         "retreat": "4/5 PULL BACK + LEVEL TOWARD THE PERSON", "approach": "5/5 APPROACH - centring the face",
         "hold": "HOLDING - waiting for the bite", "done": ""}
state = {"renderer": None, "t": time.time(), "n": 0}


def put(img, text, y, colour=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def frame(e, lines, banner=None):
    if state["renderer"] is None:
        state["renderer"] = mujoco.Renderer(e.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.40, 0.0, 0.30]
    cam.distance, cam.azimuth, cam.elevation = 1.30, 100.0, -12.0
    state["renderer"].update_scene(e.data, camera=cam)
    img = cv2.cvtColor(state["renderer"].render(), cv2.COLOR_RGB2BGR)
    # The lens is 150 deg, but MuJoCo can only draw it as a pinhole, which squashes the middle until the
    # face is a dot. Show the CENTRE of the wrist view (75 deg) instead - the same camera, cropped.
    full_fov = float(e.model.cam_fovy[e.wrist_cam])
    e.model.cam_fovy[e.wrist_cam] = 75.0
    wrist = cv2.cvtColor(e.scene.render("wrist_cam", 320), cv2.COLOR_RGB2BGR)
    e.model.cam_fovy[e.wrist_cam] = full_fov
    wrist = cv2.resize(wrist, (216, 216))
    cv2.rectangle(wrist, (0, 0), (215, 215), (0, 255, 255), 2)
    img[176:392, 416:632] = wrist            # bottom-right: the head lives top-right
    cv2.putText(img, "wrist camera (centre 75 deg)", (420, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    for k, (text, colour) in enumerate(lines):
        put(img, text, 24 + 22 * k, colour)
    if banner:
        cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 0), -1)
        put(img, banner[0], 432, banner[1], 0.7)
        if len(banner) > 2:
            put(img, banner[2], 462, (255, 255, 255), 0.55)
    return img


def show(img):
    cv2.imshow(WINDOW, img)
    state["n"] += 1
    if state["n"] % 45 == 0:
        cv2.imwrite("/tmp/feed_view.jpg", img)
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        cv2.destroyAllWindows()
        os._exit(0)
    dt = time.time() - state["t"]
    time.sleep(max(0.0, PERIOD - dt))
    state["t"] = time.time()


def tally_line(ctx):
    t = ctx["tally"]
    worst = sum(v for k, v in t.items() if k.startswith("1"))
    drop = sum(v for k, v in t.items() if k.startswith("2"))
    ok = sum(v for k, v in t.items() if k.startswith("ok"))
    return f"ep {ctx['episode']}   ok {ok}   dropped {drop}   FAST/ARM CONTACT {worst}   other {sum(t.values()) - ok - drop - worst}"


def hook(e, phase, size, o, info, ctx):
    if e.prior_hook is None:
        e.prior_hook = lambda en: show(frame(en, [("1/5 POSITION (learned, frozen)   target: " + en.name, (0, 200, 255))]))
    mode = "CALIBRATION - contact detection ON" if hf.CALIBRATE else "FEED - no contact information"
    if phase == "done":
        good = o["result"].startswith("ok")
        worst = o["result"].startswith("1")
        colour = (0, 255, 0) if good else ((0, 0, 255) if worst else (0, 165, 255))
        extra = ""
        if o.get("touch_size"):
            extra = (f"food touched: face {o['touch_size']:.1f} deg wide = {100*o['touch_size']/hf.FISHEYE_H_FOV:.0f}% of image "
                     f"width, at {o['touch_speed']:.3f} m/s")
        elif o.get("stop_size"):
            extra = f"stopped at face {o['stop_size']:.1f} deg; food {o['ahead']:.0f} mm in front of the mouth, {o['side']:.0f} mm to the side"
        end = time.time() + 2.2
        while time.time() < end:
            show(frame(e, [(mode, (200, 200, 200)), (tally_line(ctx), (200, 200, 200))], (o["result"][2:].strip(), colour, extra)))
        return
    near = size >= ctx["contact_size"] - 8.0
    sp = o.get("closing", 0.0)
    lines = [(mode, (200, 200, 200)),
             (f"{NAMES.get(phase, phase)}   target: {o['obj']}", (0, 255, 0) if phase in ("approach", "hold") else (255, 255, 255)),
             (f"face: {size:4.1f} deg wide = {100*size/hf.FISHEYE_H_FOV:3.0f}% of image width   "
              f"(touch ~{ctx['contact_size']:.0f} deg" + ("" if hf.CALIBRATE else f", stop {ctx['stop_size']:.0f}") + ")"
              if size > 0 else "face: not in view", (0, 255, 255)),
             (f"closing speed on the face {sp:.3f} m/s" + (f"   NEAR FACE: limit {hf.SAFE_ARRIVAL}" if near else ""),
              (0, 0, 255) if (near and sp > hf.SAFE_ARRIVAL) else (0, 200, 120)),
             (tally_line(ctx), (200, 200, 200))]
    show(frame(e, lines))


def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1120, 840)
    hf.STEP_HOOK = hook
    hf.main(n_episodes=10 ** 6, report=False)


if __name__ == "__main__":
    main()

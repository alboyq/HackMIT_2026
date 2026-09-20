"""'Feed me the strawberry' -> where is it, how wide is it, which grasp class is it.

Open-vocabulary detection (YOLO-World): the food is named in plain words at run time, nothing is trained per
food and nothing assumes a background. Distractor names are added alongside the target so the detector has
somewhere else to put the hand, the cup and the robot's own claws.

Geometry: one camera cannot measure depth, so - exactly as the sim's noise model assumes - the object is
taken to be resting on the TABLE. Its position is where the ray through the box centre meets the plane half
an object-height above the table; its width is the box's ANGULAR width times that range. This needs the
camera pose in the robot base frame: T_base_cam = FK(joint angles) x T_tcp_cam (handeye.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera_model import CameraModel

# which of the three trained grasp classes a food behaves like (one-hot slot in the policy observation)
SLOT = {"apple": "apple", "strawberry": "apple", "grape": "apple", "orange": "apple", "tomato": "apple", "plum": "apple",
        "meatball": "apple", "donut hole": "apple", "egg": "apple",
        "mug": "mug", "cup": "mug", "can": "mug", "bottle": "mug", "banana": "mug", "carrot": "mug", "sausage": "mug",
        "block": "block", "cracker": "block", "cheese": "block", "cheese cube": "block", "brownie": "block",
        "sandwich": "block", "cookie": "block", "tofu": "block", "bread": "block", "sushi": "block"}
PEOPLE = ["human face", "person"]        # reported separately: they CONFIRM the face sensor (see face_sensor.confirm)
DISTRACTORS = ["hand", "robot gripper", "table", "plate", "laptop", "phone", "cable"]


MENU = ["strawberry", "grape", "can"]        # what is on the table today; edit or pass --menu
RED_FRUIT = {"grape", "strawberry", "cherry", "raspberry", "tomato"}


_CLIP = {}


def _clip_probs(crops_bgr, labels):
    """Zero-shot CLIP (ViT-B/32, already installed for YOLO-World) on tight crops. Measured on this camera: strawberry
    0.84-0.94, grape 1.00 on ~50 px fruit that YOLO-World scores at 0.01-0.07."""
    import clip
    import cv2
    import torch
    from PIL import Image
    if "m" not in _CLIP:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP["m"], _CLIP["prep"] = clip.load("ViT-B/32", device=dev); _CLIP["dev"] = dev
    key = tuple(labels)
    if _CLIP.get("key") != key:
        with torch.no_grad():
            T = _CLIP["m"].encode_text(clip.tokenize(["a close-up photo of a " + l for l in labels]).to(_CLIP["dev"]))
        _CLIP["T"], _CLIP["key"] = T / T.norm(dim=-1, keepdim=True), key
    ims = torch.stack([_CLIP["prep"](Image.fromarray(cv2.cvtColor(cv2.resize(c, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))) for c in crops_bgr])
    with torch.no_grad():
        I = _CLIP["m"].encode_image(ims.to(_CLIP["dev"])); I = I / I.norm(dim=-1, keepdim=True)
        return (100 * I @ _CLIP["T"].T).softmax(-1).float().cpu().numpy()


def red_fruit_box(frame_bgr, target):
    """Small red fruit on the table: colour proposes, CLIP decides.
      1. saturated-red blobs of a plausible size,
      2. kept only if what SURROUNDS them is table (pale, unsaturated) - a brown wooden wall passes a red threshold
         and CLIP happily calls a patch of it 'grape',
      3. each survivor is cropped tight, enlarged, and classified strawberry / grape / cherry tomato by CLIP.
    Returns the box of the candidate most confidently the target (>= 0.6), else None."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 40), (12, 255, 255)) | cv2.inRange(hsv, (160, 90, 40), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(red)
    H, W = red.shape
    table = (hsv[..., 1] < 70) & (hsv[..., 2] > 110)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (250 < a < 0.05 * H * W) or w > 3 * h or h > 3 * w:
            continue
        m = int(0.5 * max(w, h))
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        ring = np.ones((y1 - y0, x1 - x0), bool); ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        if table[y0:y1, x0:x1][ring].mean() < 0.6:
            continue
        m2 = int(0.35 * max(w, h))
        cands.append(((float(x), float(y), float(x + w), float(y + h)), frame_bgr[max(0, y - m2):y + h + m2, max(0, x - m2):x + w + m2]))
    if not cands:
        return None
    # Only what is actually on the table competes. With every red fruit as a label the strawberry came out 'raspberry'
    # (0.72) and the grape 'cherry' (0.96); between just strawberry and grape they score 0.84-0.94 and 1.00.
    labels = sorted(set(MENU) & RED_FRUIT | {target})
    p = _clip_probs([c for _, c in cands], labels)[:, labels.index(target)]
    k = int(np.argmax(p))
    return cands[k][0] if p[k] >= 0.6 else None


@dataclass
class ObjectReading:
    seen: bool
    label: str = ""
    conf: float = 0.0
    box: tuple = None                   # x1, y1, x2, y2 pixels
    slot: str = "apple"                 # apple | mug | block
    width_deg: float = 0.0
    position: np.ndarray = None         # metres, robot base frame (None without a camera pose)
    width_m: float = None
    people: list = None                 # [(x1, y1, x2, y2, conf)] of every human face / person seen in the same pass


class ObjectSensor:
    def __init__(self, cam: CameraModel, weights: str = "yolov8s-worldv2.pt", conf: float = 0.08):
        from ultralytics import YOLOWorld
        self.model, self.cam, self.conf, self.target = YOLOWorld(weights), cam, conf, None

    def set_food(self, name: str, extra: list | None = None):
        self.target = name.strip().lower()
        self.names = [self.target] + [d for d in PEOPLE + DISTRACTORS + (extra or []) if d != self.target]
        self.model.set_classes(self.names)

    def read(self, frame_bgr, T_base_cam: np.ndarray | None = None, table_z: float = 0.0) -> ObjectReading:
        assert self.target, "call set_food('strawberry') first"
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        best, people = None, []
        for b in res.boxes:
            if int(b.cls[0]) == 0 and (best is None or float(b.conf[0]) > float(best.conf[0])):
                best = b
            if self.names[int(b.cls[0])] in PEOPLE:
                people.append(tuple(float(v) for v in b.xyxy[0]) + (float(b.conf[0]),))
        box, conf = (None, 0.0) if best is None else (tuple(float(v) for v in best.xyxy[0]), float(best.conf[0]))
        if self.target in RED_FRUIT and conf < 0.25:
            # Measured on the real wrist camera: a grape / strawberry ~40 px across scores 0.01-0.07 with YOLO-World, i.e.
            # nothing. On a plain table they are the only saturated red things in view, so colour finds them every frame.
            alt = red_fruit_box(frame_bgr, self.target)
            if alt is not None:
                box, conf = alt, max(conf, 0.5)
        if box is None:
            return ObjectReading(False, people=people)
        x1, y1, x2, y2 = box
        cy = 0.5 * (y1 + y2)
        r = ObjectReading(True, self.target, conf, (x1, y1, x2, y2), people=people)
        r.width_deg = self.cam.angle_between((x1, cy), (x2, cy))
        aspect = (y2 - y1) / max(1.0, x2 - x1)
        r.slot = SLOT.get(self.target, "mug" if aspect > 1.35 else "apple")
        if T_base_cam is not None:
            ray_c = self.cam.rays([(0.5 * (x1 + x2), cy)])[0]
            R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
            ray = R @ ray_c
            if ray[2] < -1e-3:                                     # looking down at the table
                # first pass: hit the table; second pass: hit half a width above it (objects ~ as tall as wide)
                rng = (table_z - t[2]) / ray[2]
                w = 2.0 * rng * np.tan(np.radians(r.width_deg) / 2.0)
                rng = (table_z + 0.5 * w - t[2]) / ray[2]
                r.position, r.width_m = t + rng * ray, float(2.0 * rng * np.tan(np.radians(r.width_deg) / 2.0))
        return r

    @staticmethod
    def draw(frame, r: ObjectReading):
        import cv2
        if not r.seen:
            cv2.putText(frame, "food not found", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        x1, y1, x2, y2 = map(int, r.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        t = f"{r.label} {r.conf:.2f}  {r.width_deg:.1f} deg  grasp class: {r.slot}"
        if r.width_m:
            t += f"  {100*r.width_m:.1f} cm"
        cv2.putText(frame, t, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, t, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

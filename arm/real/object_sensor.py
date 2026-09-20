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
        if best is None:
            return ObjectReading(False, people=people)
        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        cy = 0.5 * (y1 + y2)
        r = ObjectReading(True, self.target, float(best.conf[0]), (x1, y1, x2, y2), people=people)
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

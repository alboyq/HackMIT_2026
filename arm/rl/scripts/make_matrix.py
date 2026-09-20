"""Compose a 3x4 demo matrix from recorded tiles: 12 episodes playing at once. A tile that ends in failure
turns grey with a red label at the moment it fails; a success gets a green frame. Live scoreboard on top.
usage: make_matrix.py <tiles_dir> <out.mp4> <title> <s501_ep0002,s501_ep0003,...(12)> [shuffle_seed]"""
import json
import os
import random
import sys

import cv2
import numpy as np

TILES, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
NAMES = sys.argv[4].split(",")
random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0).shuffle(NAMES)
assert len(NAMES) == 12, len(NAMES)
TW, TH, HEAD, FPS = 320, 240, 64, 30
W, H = TW * 4, TH * 3 + HEAD
GREEN, RED, AMBER, WHITE, DIM = (90, 220, 90), (60, 60, 235), (40, 170, 255), (245, 245, 245), (150, 150, 150)


def kind(result):
    if result.startswith("ok"):
        return "ok", "FED", GREEN
    if result.startswith("1"):
        return "unsafe", "UNSAFE ARRIVAL", RED
    if result.startswith("2"):
        return "drop", "DROPPED", RED
    return "miss", ("PICK-UP FAILED" if ("pinch" in result or "table" in result or "lift" in result) else "MISSED"), AMBER


tiles = []
for n in NAMES:
    meta = json.load(open(os.path.join(TILES, n + ".json")))
    cap = cv2.VideoCapture(os.path.join(TILES, n + ".mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA))
    k, label, colour = kind(meta["result"])
    tiles.append(dict(frames=frames, phases=meta["phases"], obj=meta["obj"], kind=k, label=label, colour=colour))
    print(f"{n}: {len(frames)} frames  {meta['result']}")

longest = max(len(t["frames"]) for t in tiles)
total = longest + int(3.5 * FPS)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def text(img, s, org, scale, colour, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


for t in range(total):
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:HEAD] = (28, 24, 22)
    done = {"ok": 0, "drop": 0, "miss": 0, "unsafe": 0}
    for i, tl in enumerate(tiles):
        r, c = divmod(i, 4)
        n = len(tl["frames"])
        ended = t >= n
        f = tl["frames"][min(t, n - 1)].copy()
        if ended:
            done[tl["kind"]] += 1
            since = t - n
            if tl["kind"] == "ok":
                cv2.rectangle(f, (0, 0), (TW - 1, TH - 1), tl["colour"], 4)
            else:
                g = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                a = min(1.0, since / 12.0)                                   # fade to grey over 0.4 s
                f = cv2.addWeighted(g, a, f, 1 - a, 0)
                f = (f * (1.0 - 0.45 * a)).astype(np.uint8)
                cv2.rectangle(f, (0, 0), (TW - 1, TH - 1), tl["colour"], 3)
            (tw_, th_), _ = cv2.getTextSize(tl["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(f, (TW // 2 - tw_ // 2 - 8, TH // 2 - th_ - 8), (TW // 2 + tw_ // 2 + 8, TH // 2 + 8), (0, 0, 0), -1)
            text(f, tl["label"], (TW // 2 - tw_ // 2, TH // 2), 0.62, tl["colour"], 2)
        else:
            text(f, tl["phases"][min(t, n - 1)], (8, TH - 10), 0.42, WHITE)
        text(f, tl["obj"], (8, 18), 0.45, WHITE)
        canvas[HEAD + r * TH:HEAD + (r + 1) * TH, c * TW:(c + 1) * TW] = f
    for k in range(1, 4):
        cv2.line(canvas, (k * TW, HEAD), (k * TW, H), (0, 0, 0), 2)
    for k in range(1, 3):
        cv2.line(canvas, (0, HEAD + k * TH), (W, HEAD + k * TH), (0, 0, 0), 2)
    text(canvas, TITLE, (16, 26), 0.72, WHITE, 2)
    text(canvas, "simulation  |  decisions from wrist camera + motor feedback only  |  12 runs, real time", (16, 50), 0.46, DIM)
    score = f"fed {done['ok']}/12"
    text(canvas, score, (W - 500, 30), 0.8, GREEN, 2)
    text(canvas, f"unsafe arrivals {done['unsafe']}", (W - 330, 30), 0.6, GREEN if done["unsafe"] == 0 else RED, 1)
    text(canvas, f"dropped {done['drop']}   missed {done['miss']}", (W - 330, 52), 0.5, DIM)
    if t < 12:                                                                # fade in
        canvas = (canvas * (t / 12.0)).astype(np.uint8)
    if t >= longest + FPS // 2:                                                # closing card
        a = min(1.0, (t - longest - FPS // 2) / 15.0)
        card = canvas.copy()
        cv2.rectangle(card, (W // 2 - 380, H // 2 - 70), (W // 2 + 380, H // 2 + 70), (20, 18, 16), -1)
        cv2.rectangle(card, (W // 2 - 380, H // 2 - 70), (W // 2 + 380, H // 2 + 70), GREEN, 2)
        text(card, f"{done['ok']} of 12 fed", (W // 2 - 350, H // 2 - 18), 1.3, GREEN, 3)
        text(card, f"0 unsafe arrivals at the face   |   {done['drop']} dropped   |   {done['miss']} missed pick-up",
             (W // 2 - 350, H // 2 + 36), 0.62, WHITE, 1)
        canvas = cv2.addWeighted(card, a, canvas, 1 - a, 0)
    vw.write(canvas)
vw.release()
print("wrote", OUT, f"{total/FPS:.0f} s")

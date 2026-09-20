import sys, cv2, numpy as np
out, srcs = sys.argv[1], sys.argv[2:]
vw, last = None, None
for k, s in enumerate(srcs):
    cap = cv2.VideoCapture(s)
    w, h = int(cap.get(3)), int(cap.get(4))
    if vw is None:
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if last is not None and n < 15:                      # half-second crossfade between the three
            f = cv2.addWeighted(f, n / 15.0, last, 1 - n / 15.0, 0)
        vw.write(f); n += 1; cur = f
    last = cur
    print(s, n, "frames")
vw.release(); print("wrote", out)

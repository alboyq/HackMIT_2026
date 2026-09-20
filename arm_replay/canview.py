"""Decode a canlog.py capture: MIT commands (id 1-6) and motor replies (id 0x11-0x16)."""
import sys, numpy as np
SPEC = {1:(12.5,10,28),2:(12.5,10,28),3:(12.5,10,28),4:(12.5,30,10),5:(12.5,30,10),6:(12.5,30,10)}     # p_max, v_max, t_max
ERR = {0:"disabled",1:"normal",8:"overvolt",9:"undervolt",10:"overcurrent",11:"MOS overtemp",12:"coil overtemp",13:"comm loss",14:"overload"}
u2f = lambda u, lo, hi, b: u * (hi - lo) / ((1 << b) - 1) + lo
def load(path):
    cmds, fb, other = {i: [] for i in range(1, 7)}, {i: [] for i in range(1, 7)}, 0
    for line in open(path):
        t, i, h = line.split(); t, i, d = float(t), int(i, 16), bytes.fromhex(h)
        if 1 <= i <= 6 and len(d) == 8:
            if d[:7] == b"\xff" * 7: cmds[i].append((t, "special", d[7])); continue
            pm, vm, tm = SPEC[i]; p = (d[0] << 8) | d[1]; v = (d[2] << 4) | (d[3] >> 4); kp = ((d[3] & 0xF) << 8) | d[4]; kd = (d[5] << 4) | (d[6] >> 4); tq = ((d[6] & 0xF) << 8) | d[7]
            cmds[i].append((t, "mit", u2f(p, -pm, pm, 16), u2f(v, -vm, vm, 12), u2f(kp, 0, 500, 12), u2f(kd, 0, 5, 12), u2f(tq, -tm, tm, 12)))
        elif 0x11 <= i <= 0x16 and len(d) == 8:
            m = i - 0x10; pm, vm, tm = SPEC[m]
            fb[m].append((t, ERR.get(d[0] >> 4, "?"), u2f((d[1] << 8) | d[2], -pm, pm, 16), u2f((d[3] << 4) | (d[4] >> 4), -vm, vm, 12), u2f(((d[4] & 0xF) << 8) | d[5], -tm, tm, 12)))
        else: other += 1
    return cmds, fb, other
if __name__ == "__main__":
    c, f, o = load(sys.argv[1]); print("commands per motor:", {k: len(v) for k, v in c.items()}); print("replies  per motor:", {k: len(v) for k, v in f.items()}, " other frames:", o)
    for m in (1, 3): 
        mit = [x for x in c[m] if x[1] == "mit"]
        if mit: print(f"  motor {m}: first MIT command kp={mit[0][4]:.1f} kd={mit[0][5]:.2f}  |  first reply state={f[m][0][1] if f[m] else None}")

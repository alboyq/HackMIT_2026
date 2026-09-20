"""PASSIVE CAN listener: receives and timestamps every frame on can0, sends NOTHING.
    python canlog.py out.txt      (stop with SIGTERM/Ctrl-C; prints the frame count)"""
import can, signal, sys
bus = can.Bus(interface="socketcan", channel="can0")
out, stop, n = open(sys.argv[1], "w"), {"f": False}, 0
for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, lambda *a: stop.__setitem__("f", True))
while not stop["f"]:
    m = bus.recv(timeout=0.3)
    if m is None: continue
    out.write(f"{m.timestamp:.6f} {m.arbitration_id:X} {bytes(m.data).hex()}\n"); n += 1
out.close(); bus.shutdown(); print("frames:", n)

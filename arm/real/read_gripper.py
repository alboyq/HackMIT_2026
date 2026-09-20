"""Print gripper motor 0x08's position reading. READ-ONLY: nothing is enabled; the single frame sent has zero
gains and zero torque and goes to the gripper motor alone. Refuses to run while anything else is driving the arm.
    push the jaws fully CLOSED by hand -> run -> note the number;  fully OPEN -> run again.
    python arm/real/read_gripper.py [label]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

busy = subprocess.run(["pgrep", "-af", "python[^ ]* .*(replay_pose|real_arm.py|-m openyam)"], capture_output=True, text=True).stdout.strip()
if busy:
    raise SystemExit("something is driving the arm right now - not touching the bus:\n" + busy)
sys.path.insert(0, str(Path.home() / "openyam"))
from openyam.arm import YAM_GRIPPER, Motor  # noqa: E402
from openyam.gsusb import CanBus  # noqa: E402

label = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
bus = CanBus()
try:
    m = Motor(bus, *YAM_GRIPPER)
    reads = []
    for _ in range(5):
        st = m.read_state()
        if st is not None:
            reads.append(st)
        time.sleep(0.05)
    if not reads:
        raise SystemExit("gripper 0x08: no reply")
    pos = [r.position for r in reads]
    print(f"gripper 0x08 [{label}]: {sum(pos)/len(pos):+.4f} rad  (spread {max(pos)-min(pos):.4f}, state '{reads[-1].error}', {len(reads)} replies)")
    out = Path(__file__).resolve().parent / "calibration" / "gripper_raw.json"
    out.parent.mkdir(exist_ok=True)
    d = json.loads(out.read_text()) if out.exists() else {}
    d[label] = {"raw_rad": sum(pos) / len(pos), "state": reads[-1].error, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.write_text(json.dumps(d, indent=2))
finally:
    bus.close()

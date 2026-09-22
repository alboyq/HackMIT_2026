"""Bridges a `decoder.selected`-style object pick to the real arm.

The arm (OpenYAM, on the shared GX10 compute) is driven by
`arm_teach_replay/replay.py` (teach-and-replay executor, see
Teach_and_Replay_Arm_PRD.md) over SSH -- there is no local driver here, this
module just shells out to the same non-interactive SSH path used throughout
this project (`ssh -J jump@... asus@...`) and turns the script's own stdout
into `arm.status` broadcasts the frontend pages listen for.

Only one delivery runs at a time; a request that arrives while one is already
in flight gets an immediate `BUSY` status instead of queuing (the arm is a
single physical resource with a person waiting next to it).
"""
import asyncio
import re
import time

SSH_CMD = ["ssh", "-J", "jump@129.153.206.6", "asus@10.10.10.4"]
REMOTE_CMD = (
    "cd ~/arm_teach_replay && "
    "~/openyam/.venv/bin/python replay.py object_{object_id} --to deliver --speed 0.25 --kp-scale 1.6"
)

_STAGE_RE = re.compile(r"reached (\w+):")
_MOVING_RE = re.compile(r"moving -> (\w+)")
_GUARD_RE = re.compile(r"(GUARD TRIPPED|ERROR|ABORT):?\s*(.*)")

_lock = asyncio.Lock()


async def run_arm_delivery(object_id: int, hub) -> None:
    if _lock.locked():
        await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": "BUSY",
                              "message": "arm is already mid-delivery"})
        return

    async with _lock:
        t0 = time.time()
        await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": "QUEUED"})
        cmd = SSH_CMD + [REMOTE_CMD.format(object_id=object_id)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": "FAILED",
                                  "message": "ssh not found on this machine"})
            return

        last_message = None
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            print(f"[arm] {line}", flush=True)

            m = _STAGE_RE.search(line)
            if m:
                last_message = line
                await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": m.group(1)})
                continue
            m = _MOVING_RE.search(line)
            if m:
                last_message = line
                continue
            m = _GUARD_RE.search(line)
            if m:
                last_message = m.group(2) or m.group(1)

        rc = await proc.wait()
        elapsed = time.time() - t0
        if rc == 0:
            await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": "DONE",
                                  "message": f"delivered in {elapsed:.0f}s"})
        else:
            await hub.broadcast({"type": "arm.status", "object_id": object_id, "stage": "FAILED",
                                  "message": last_message or f"replay.py exited {rc}"})

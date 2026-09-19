# Repository notes

Reviewed 2026-09-19: every Markdown/config file in the repository, the recent Git history, the current uncommitted `training/`, `skills/`, `sim/`, and tests in the shared GX10 checkout, and project-related files under `Documents`, `Downloads`, and `Desktop` (none found outside the repository). The selected repository is `https://github.com/alboyq/HackMIT_2026.git`.

## Structure and ownership

- `yam_agentic/`: teammate research and a complete simulation-only YAM scene, measured fingertip TCP, DLS IK, expert pick/present planner, tests, and rendered evidence. Treat as read-only upstream material.
- `sim/`, `skills/`, `tests/` in the shared checkout: teammate's uncommitted skill API and safety work. It provides `locate`, `grasp`, `transport`, `present`, `release`, and `home`; do not modify it.
- `training/` in the shared checkout: teammate's uncommitted ACT recorder/training path. It owns image-conditioned imitation learning and LeRobot recording; do not modify it.
- `arm/`: this branch's isolated arm-policy work. `arm/rl` owns low-dimensional MuJoCo PPO; `arm/ik` owns the no-learning demo path; shared controller selection and safety live under `arm/`.
- EEG selection and OWL-ViT detection are teammate-owned. The arm consumes their selected target and never decodes EEG or runs a second detector model.

## Existing contracts (as implemented today)

There is no implemented ZeroMQ or WebSocket contract in the repository yet. The implemented local IPC is UDP JSON:

| Flow | Transport | Address | Schema | Source |
|---|---|---|---|---|
| OWL-ViT target | UDP | `127.0.0.1:8765`, at least 5 Hz | `{"label": str, "bbox": [x1,y1,x2,y2], "image_width": int, "image_height": int, "confidence": float}` | `training/target_protocol.py`, `training/README.md` |
| Teleop target | UDP | `127.0.0.1:8766` | `{"action": [q1,q2,q3,q4,q5,q6,gripper]}`; seven finite absolute positions | `training/target_protocol.py`, `training/README.md` |

`TargetCommand.vector()` maps the detector message to `[cx, cy, width, height, confidence, valid]`, normalized by image dimensions. `LatestUDP` retains the newest valid packet. The recorder rejects a detector target older than 0.5 s and falls back to measured positions only when the optional commanded-action stream is absent.

The earlier requested ZeroMQ `target` message (`x,y,z,frame,source,confidence,timestamp`) is therefore an adapter contract, not the current team contract. New arm glue must preserve UDP/8765 and add a separate calibrated pixel-to-base adapter rather than silently replacing the teammate's schema.

## Architecture already agreed in the repo

The repository uses a four-layer boundary: agent chooses object/destination, perception outputs robot-frame poses, deterministic skills enforce safety, and control emits absolute joint targets at 30 Hz. Feeding motion stops at a staging pose and requires human confirmation for any closer approach. Delivery remains scripted; the learned policy is limited to manipulation.

The verified teammate simulation uses the MuJoCo Menagerie `i2rt_yam` model, a TCP measured from the finger-pad geoms rather than `grasp_site`, a base-mounted scene camera, and a wrist camera rigidly attached to link 6. Existing measurements and planner code are reused in `arm/ik` instead of rebuilding kinematics from scratch.

## Git/history state

- `3c53f42`: initial README.
- `144c10d`: teammate-added `yam_agentic` simulation stack and handoff (current `origin/main`).
- Work is isolated in a separate GX10 worktree on branch `arm-ik-rl`; the actively shared checkout remains on `main`.
- Generated checkpoints, datasets, logs, videos, venvs, and downloaded third-party models stay out of Git.


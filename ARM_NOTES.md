# OpenYAM arm notes

Status words: **confirmed** means a cited primary source or a read-only observation from this specific arm; **assumed** means the OpenYAM-to-I2RT mapping still needs a physical check. Conflicts are called out explicitly.

## Facts and evidence

| Fact | Status | Evidence |
|---|---|---|
| OpenYAM is Anvil Robotics' open-source desktop manipulator; core structure is CNC/sheet metal, with printed shells used only cosmetically/cable routing. The box includes a 24 V PDU, CAN/power cable, and CAN-USB adapter. | confirmed | [Anvil product page](https://shop.anvil.bot/products/openyam) |
| Anvil's public description and hardware repositories currently contain only “Coming Soon” READMEs, so they do not yet publish usable OpenYAM URDF/CAD. | confirmed | [OpenYAM_description](https://github.com/anvil-robotics/OpenYAM_description), [OpenYAM_hardware](https://github.com/anvil-robotics/OpenYAM_hardware) |
| Anvil's full software stack is Docker/ROS 2 workcell orchestration and the product page says an Anvil Devbox is required for that stack. It is not required to use the public I2RT Python SDK directly on Linux. | confirmed | [Anvil product page](https://shop.anvil.bot/products/openyam), [anvil-loader](https://github.com/anvil-robotics/anvil-loader), [loader architecture](https://deepwiki.com/anvil-robotics/anvil-loader) |
| The physical OpenYAM is mechanically/control-stack compatible with the standard I2RT YAM family (six revolute arm joints plus a parallel gripper). Exact revision is not published by Anvil. | assumed, strong | The arm on this GX10 was read-only inventoried as three DM4340 and three DM4310 arm motors; this matches the [I2RT YAM SDK/model](https://github.com/i2rt-robotics/i2rt). Anvil's public repos do not state the revision, so “YAM v1” must be verified physically. |
| Standard YAM has 6 arm DOF; SDK state/action is 6 values without a gripper or 7 with one. | confirmed | [I2RT YAM docs](https://doc.i2rt.com/products/yam) |
| The Menagerie/I2RT-derived MJCF joint limits are J1 `[-2.61799, 3.05433]`, J2/J3 `[0, 3.66519]`, J4/J5 `[-1.5708, 1.5708]`, J6 `[-2.0944, 2.0944]` radians. I2RT applies an additional safety buffer in software. | confirmed for model; physical offsets unverified | [MuJoCo Menagerie YAM MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/main/i2rt_yam/yam.xml), [I2RT YAM docs](https://doc.i2rt.com/products/yam) |
| All joints default to local `axis="0 0 1"`; child-body transforms rotate those local axes. The base/world frame is MuJoCo right-handed with +Z upward. Model zero is the MJCF zero; its provided home keyframe is `[0, 1.047, 1.047, 0, 0, 0]`. | confirmed for model | [MuJoCo Menagerie YAM MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/main/i2rt_yam/yam.xml) |
| Principal joint-origin translations are base→J1 0.0631 m, J1→J2 `(0.000025,-0.02,0.0409)`, J2→J3 0.264 m, J3→J4 `(-0.245,-0.06,0)`, J4→J5 `(-0.074,-0.0395,0.000025)`, J5→J6 `(0,0.0353,0.0395)`. These are model transforms, not simple DH link lengths. I2RT reports a 0.807 m chain length for standard YAM. | confirmed for model | [MuJoCo Menagerie YAM MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/main/i2rt_yam/yam.xml), [I2RT repository](https://github.com/i2rt-robotics/i2rt) |
| The Menagerie model's `grasp_site` is not the fingertip-center TCP. Teammate geometry tests measure the TCP from pad collision geoms; open pad separation is 82.8 mm and `grasp_site` is about 44 mm from the pad center. | confirmed in simulation | `yam_agentic/MEASUREMENTS.md` and `yam_agentic/sim/yam_ik.py`; model source: [Menagerie MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/main/i2rt_yam/yam.xml) |
| This arm's read-only inventory found CAN IDs `0x01..0x06` for the arm and `0x08` for the gripper; the latter conflicts with the vendor config's usual trailing ID. | confirmed for this arm, local observation | `/home/asus/openyam/README.md`; verify again before any future hardware session. |
| J1–J3 are DM4340 (40:1, ±28 Nm model limit) and J4–J6 are DM4310 (10:1, ±10 Nm); the observed gripper is DM4310. | confirmed for this arm, local observation | `/home/asus/openyam/README.md`; motor families and limits are also represented in the [I2RT model](https://github.com/google-deepmind/mujoco_menagerie/blob/main/i2rt_yam/yam.xml). |
| CAN uses 1 Mbit/s. I2RT uses SocketCAN (`can0`) on Linux and supports position commands with PD tracking, joint-state commands, gravity compensation, and sim mode. | confirmed | [I2RT motors](https://doc.i2rt.com/products/motors), [I2RT SDK](https://github.com/i2rt-robotics/i2rt), [YAM docs](https://doc.i2rt.com/products/yam) |
| Factory I2RT documentation says a 400 ms timeout enters damping mode after lost commands; it warns that disabling this while gravity compensation is active can leave uncontrolled torque. | confirmed vendor guidance | [I2RT SDK safety timeout](https://github.com/i2rt-robotics/i2rt) |
| Local pre-hack testing says this specific arm reported `TIMEOUT=0` and the local driver warns that changing timeout behavior mid-session caused a fault/re-enable cascade. This conflicts with the current I2RT factory-default guidance. | confirmed conflict | `/home/asus/openyam/README.md` versus [I2RT SDK](https://github.com/i2rt-robotics/i2rt). Do not change the register during the hackathon; verify with Anvil/I2RT before future use. |
| Motor zeroing is persistent and available per motor. It must not be run merely to “fix” a pose mismatch. | confirmed | [I2RT SDK set-zero tool](https://github.com/i2rt-robotics/i2rt) |
| Public I2RT gripper models/configs include `linear_4310`, `linear_3507`, `flexible_4310`, `crank_4310`, no-gripper, and the YAM teaching handle. | confirmed | [I2RT SDK repository](https://github.com/i2rt-robotics/i2rt) |
| The standard `linear_4310` config uses a DM4310, needs calibration unless limits are overridden, and the I2RT factory clamps gripper effort. | confirmed | [I2RT SDK](https://github.com/i2rt-robotics/i2rt) |
| The I2RT SDK installs on Linux/aarch64 as Python and drives SocketCAN directly; GELLO also lists I2RT YAM support. Anvil's proprietary workcell path uses its Devbox/Docker stack, but it is not the only safe Python route. | confirmed | [I2RT SDK](https://github.com/i2rt-robotics/i2rt), [GELLO](https://github.com/wuphilipp/gello_software), [Anvil loader](https://github.com/anvil-robotics/anvil-loader) |

## Model choice

Use `google-deepmind/mujoco_menagerie/i2rt_yam/yam.xml` for the shared simulation and IK model. It is single-arm, already contains the parallel gripper and position actuators, documents its derivation from the public I2RT URDF, and is the exact model the teammate's tested `yam_agentic` stack uses. The I2RT repository also provides URDF and modular arm/station MJCFs, but Anvil's OpenYAM description repo does not yet provide a model to compare. The model/action decision is therefore 6 arm joints plus one gripper action, 30 Hz control to match the repository's proven skill layer, and conservative simulated caps pending physical verification.

## Control and safety

- Default is simulation or dry-run. No module may import or instantiate the real driver without `--enable-hardware`.
- Never change motor IDs, zero offsets, timeouts, control mode, PMAX/VMAX/TMAX, drivers, kernel, or CAN setup as part of policy development.
- Before any later hardware session: identify the exact YAM revision and gripper; read joint state and live config; reconcile signs/offsets with FK; confirm every limit with at least 0.15 rad margin; seed commands from measured pose; cap joint speed initially at 0.25 rad/s; keep a human on E-stop.
- Do not disable or remove power from an unsupported raised arm. Conversely, do not assume the current hold behavior is safe merely because power remains on. A human must control the hardware-state transition.
- No learned controller enters the mouth keep-out zone. Scripted delivery stops at staging; closer motion requires explicit human approval and a 0.05 m/s Cartesian cap.

## Verify on the physical arm (no action while owner is away)

- Photograph/record the arm revision label and every motor/gripper model.
- Confirm CAN IDs, bitrate, control mode, timeout register, and gripper polarity/range read-only.
- ~~Measure model-vs-hardware joint signs and zero offsets one joint at a time while unpowered/safely supported.~~ **DONE 2026-09-19** — see "Measured joint map" below and `joint_map_measured.json`.
- Compare SDK FK and model FK at at least five static poses; require TCP error under 10 mm before Cartesian commands. **STILL OUTSTANDING — this is the remaining gate before `RealArm` gets a backend.**
- Measure actual gripper opening, calibration endpoints, safe force setting, and whether the wrist camera changes the end-effector mass.
- Confirm base frame orientation, table height, reachable workspace, mechanical stops, E-stop behavior, and the correct procedure for safely lowering while powered.

## Measured joint map (2026-09-19)

Hand-measured on the real arm. **Motors were disabled for the entire session** — every frame sent
was a zero MIT command (`kp=kd=torque=0`) and `enable()` was never called, so nothing was ever
energised. Signs came from moving one joint by hand and reading the encoder delta; offsets from
both mechanical hard stops per joint. Full data, per-joint stop values and method in
[`joint_map_measured.json`](joint_map_measured.json).

**Convention: `q_model = sign * q_encoder - offset` (radians).**

| joint | motor | sign | offset (rad) | offset (deg) | range vs MJCF | confidence |
|---|---|---|---|---|---|---|
| J1 | `0x01` | +1 | **-1.4313** | **-82.00** | -2.1 deg | high |
| J2 | `0x02` | +1 | +0.0044 | +0.25 | +1.5 deg | high |
| J3 | `0x03` | +1 | +0.0101 | +0.58 | **+16.1 deg** | **see caveat** |
| J4 | `0x04` | +1 | -0.0074 | -0.43 | +10.8 deg | high |
| J5 | `0x05` | +1 | +0.0015 | +0.09 | +0.8 deg | high |
| J6 | `0x06` | +1 | **+1.3035** | **+74.69** | -1.0 deg | high |

**All six signs are +1.** Two joints carry large zero offsets: **J1 (base yaw, -82 deg)** and
**J6 (gripper roll, +75 deg)**. Commanding MJCF joint values straight to this hardware would point
the whole arm ~82 deg off heading with the jaws rolled ~75 deg — at opposite ends of the chain,
and invisible in simulation. Do not skip the offsets.

### Possible error — J3 is not fully pinned

J3's measured range is **16.1 deg wider** than the MJCF's, so its two hard stops do not determine
the offset arithmetically. Two readings are consistent with the measurements:

- **`offset = +0.0101`** (adopted) — assumes the lower stop maps to the model minimum. J2 is the
  same joint family with the same model range and behaves exactly this way, landing 0.25 deg from
  the model zero; J3 lands 0.58 deg from it. The extra 16 deg would then all sit at the top of the
  range, which is where a removed I2RT software safety buffer would appear.
- **`offset = +0.1509` (+8.65 deg)** — assumes the midpoints align. This would require J3's zero to
  be displaced in a way no other joint on this arm is.

The FK check separates them cleanly: an 8.65 deg elbow error is roughly **4 cm** at the gripper.
Treat J3's offset as provisional until then. The low stop was re-verified by hand and did not move
under firm pressure.

Ranges exceeding the MJCF on J2/J3/J4/J5 are expected rather than anomalous — this file already
records that I2RT applies a software safety buffer, and that this Anvil OpenYAM is of an
unpublished revision relative to the standard I2RT YAM the MJCF derives from. A wider real range is
the safe direction: the planner uses MJCF limits and never asks for the extra travel.

### Still to complete before `RealArm` gets a backend

1. **FK check at five static poses, TCP error under 10 mm.** The remaining gate. Also resolves J3.
2. **Gripper pad force.** `0x08` reads `TMAX = 10.0 Nm`, but that is motor torque and the scaling
   for MIT torque commands — not pad force. Converting needs the `linear_4310` rack-and-pinion
   geometry (meshes vendored at `third_party/i2rt/i2rt/robot_models/gripper/`). The sim's `kp=800`
   (~10 N/pad) remains unvalidated against hardware.
3. **Confirm base frame orientation, table height and reachable workspace** against the measured map.
4. **Secure the CAN adapter.** It re-enumerated **three times** during this session (USB device
   005 -> 006 -> 007). With `TIMEOUT=0` a motor holds its last command forever if the host stops
   talking, and a USB disconnect is the one failure where disable-on-exit cannot run — the host
   cannot send the disable frame on a bus that no longer exists. This is harmless with motors
   disabled and unacceptable once anything is energised.

## Open questions

- Which exact I2RT YAM revision did Anvil use for this OpenYAM build?
- Is the gripper Anvil-specific or the I2RT linear_4310, and why is it on CAN ID `0x08`?
- Which timeout policy does Anvil recommend for this firmware/build, given the conflict between current I2RT docs and local incident notes?
- What payload/tool inertia should be configured for the actual gripper, wrist camera, and food utensil?


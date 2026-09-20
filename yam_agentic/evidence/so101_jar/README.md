# Evidence from the SO-101 jar work ("so101Sim")

The plan page and `GAME_PLAN.md` quote measurements from a project that is **not in this
repository**: `so101Sim`, earlier sim-to-real work by one teammate on an SO-101 arm, living on
their laptop. It is a MuJoCo pipeline that generates demonstrations with a scripted expert,
composites the rendered arm and object over real photos, trains ACT on the result, and evaluates
closed-loop in simulation. The task there was lifting a spice jar with a side grasp, seen from one
fixed third-person camera.

It matters here because the sim-first plan for the YAM is the same recipe on a new arm, and these
are the only numbers the team has about how that recipe behaves.

**These are reference copies.** They contain absolute paths to that laptop and depend on parts of
`so101Sim` that are not copied (the SO-101 MJCF scenes, the lift environment, a virtualenv), so
they do not run from here. They are here so that every number can be traced to the code and the
output that produced it. The YAM code that *does* run from this repo is in `../../sim/`.

## Where each claim comes from

| Claim on the plan page | File |
|---|---|
| Best sim-trained policy: 83–86 % closed-loop (`jar_act_v3/step_15000`) | `results/jar_act_v3.results.txt`; scoreboard in `docs/ARRIVAL_DAY.md` §6 |
| In-training backgrounds re-run: 84 % kitchen, 83 % wallpaper pool | `results/jar_act_v3.stress_obs.txt` (row `clean`), `results/jar_act_v3.results.txt` |
| Held-out backgrounds: 81 % then 70 % | `bg_heldout/README.md`; row `heldout30` in `results/jar_act_v3.stress_obs.txt` |
| "never-seen backgrounds" was mislabelled | correction note in `docs/ARRIVAL_DAY.md` §6; the label was printed by `code/run_chain_fast.sh` |
| Camera-realism stress test | `code/obs_corrupt.py`, `code/stress_test_obs.sh`, `results/jar_act_v3.stress_obs.txt`, `results/camera_corruptions_contact_sheet.jpg` |
| Camera position is the fragile axis: 84 / 66 / 46 % at ±2 / 7 / 10 cm | `docs/ARRIVAL_DAY.md` §6 (overnight log) |
| Wider camera randomisation: 83 / 77 / 69 / 77 / 58 % at ±0 / 4 / 10 / 15 / 20 cm | `results/act_widecam_filtered.results.txt` (compare `results/act_widecam.results.txt`) |
| Checkpoints swing 46 / 67 / 44 / 56 % | `results/jar_act_v2.results.txt` |
| Lower camera elevation: 58–64 % | `results/jar_act_v2b.results.txt` |
| What training randomises (and what it does not) | `code/jar_scene.py`, `reset()` and `_render_now()` |
| How demonstrations are generated and the policy trained | `code/gen_demos.py`, `code/scripted_expert_side.py`, `code/train_act_jar.py`, `code/run_chain_fast.sh` |

## Conditions behind every number

96 episodes per row unless stated, held-out seeds (9,500,000+), success = jar lifted more than
5 cm, upright, held 0.4 s. Scene settings match training: camera lag 1–3 control ticks, a cast
shadow under the jar, servo gain scaled 0.1–1×, friction 0.35–0.7, jar zone r 0.31–0.41 m. The
binomial error at n = 96 is about ±4 points, so single rows that differ by less than that are
not evidence of anything; rows in the stress test share seeds, which makes differences between
them tighter than that.

## What these numbers are not

Every one is simulation evaluated against simulation. The background is a flat photo behind a
rendered arm; rendered lighting does not match the photo. No policy from `so101Sim` has ever
been given a real camera frame or run on a real arm. They say how the recipe responds to
backgrounds, camera effects and camera placement *in sim* — which is what should shape the
YAM data generator — and nothing about real-world success.

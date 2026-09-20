# bg_heldout — backgrounds no policy has ever trained on

30 natural photographs used ONLY for evaluation. They exist because the project's earlier
"never-seen backgrounds" number was not what it said: that evaluation (`--p-real 0.0`) drew from
`rl/jar/bg_pool`, the same 136-wallpaper pool that supplies half of every training set. It
measured "not the kitchen photo", not "unseen". This directory is the real held-out set.

**Rule: never pass this directory to `gen_demos.py` or use it as a `bg_pool`.** `gen_demos.py`
refuses it. If an image here ever leaks into training, replace the set and re-baseline.

Provenance: `https://picsum.photos/seed/yamheld{1..30}/1600/900`, fetched 2026-09-19.
The office-desk photo (`rl/real/bg_desk_view1_stale`) was part of the first held-out run for
`jar_act_v3` but is deliberately NOT in this set: the desk-era policies (`wide_cam_demos*`,
`post_act_desk1*`) trained on it, so it is only held out for some checkpoints.

## Use

```bash
cd urlab_bridge
# export the run's scene env first (OBJECT_SPEC JAR_ZONE KP_SCALE JAR_FRICTION OBS_LATENCY JAR_SHADOW)
.venv/bin/python ../rl/jar/eval_act_jar.py CKPT 96 --calib CALIB --bg ../rl/jar/bg_heldout \
    --na 25 --p-real 1.0 --seed0 9500000 --workers 14
```

`--p-real 1.0` makes every episode use a photo from `--bg`, i.e. 100 % held-out backgrounds.
`rl/jar/stress_test_obs.sh` runs this together with the camera-corruption conditions.

## Baselines

| checkpoint | in-training backgrounds | held-out | note |
|---|---|---|---|
| `jar_act_v3/act/step_15000` | 84 % kitchen photos, 83 % wallpaper pool | 81 % (78/96), then **70 %** (67/96) | 2026-09-19. First run used 31 images (this set + the office-desk photo); second run used this set alone on the same scenes. Same seeds, different photo-to-episode assignment. Pooled 76 %: a real gap of several points, sensitive to which photos come up. Larger run pending. |

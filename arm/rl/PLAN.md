# Plan

## Running now

1. Train PPO reach with the I2RT YAM model, 12 CPU environments, and VecNormalize.
2. Monitor throughput, success, RAM, and system load; reduce workers if the shared machine is affected.

## Next

1. Add food proxies and keep the existing 23-value observation and 7-value action spaces.
2. Validate contact physics with the scripted damped-least-squares IK baseline.
3. Add grasp, then lift curricula, warm-starting from the previous checkpoint.
4. Add evaluation and rollout MP4 export.
5. Only after training is running: ZeroMQ target input, fake publisher, camera sources, and calibration tools.
6. Add dry-run deployment with hardware gated behind `--enable-hardware`.


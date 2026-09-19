from pathlib import Path

import numpy as np

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMReachEnv


ROOT = Path(__file__).resolve().parents[1]


def test_model_loads_and_env_steps():
    env = OpenYAMReachEnv(load_config(ROOT / "configs/default.yaml"))
    try:
        obs, info = env.reset(seed=1)
        assert obs.shape == (23,)
        assert env.action_space.shape == (7,)
        assert np.isfinite(obs).all()
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        assert obs.shape == (23,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_actions_are_clipped():
    env = OpenYAMReachEnv(load_config(ROOT / "configs/default.yaml"))
    try:
        env.reset(seed=2)
        env.step(np.full(7, 100.0, dtype=np.float32))
        assert np.max(np.abs(env.previous_action)) <= 1.0
    finally:
        env.close()


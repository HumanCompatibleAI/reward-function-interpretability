"""Sparse versions of Mujoco environments."""

import gym
from gym.envs.mujoco import reacher
import numpy as np


class SparseReacher(reacher.ReacherEnv):
    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward = (np.linalg.norm(vec) < 0.05).astype(np.float32)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}


gym.register(
    "imitation/SparseReacher-v0",
    entry_point="imitation.envs.sparse:SparseReacher",
    # as in original Mujoco environment
    max_episode_steps=50,
)

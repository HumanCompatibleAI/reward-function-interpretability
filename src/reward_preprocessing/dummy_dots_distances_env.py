"""Dummy 'environment' for dots and distance data (see scripts/gen_dots_and_dists.py)"""

import gym
from gym import spaces
import numpy as np


class DotsAndDistsEnv(gym.Env):
    """Dummy environment: no methods implemented, only obs_space and act_space set."""

    def __init__(self, size: int):
        # Note that size must be specified in pixels
        super().__init__()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(1)

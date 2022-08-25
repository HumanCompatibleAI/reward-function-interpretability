import logging
from typing import List, Tuple, cast

import gym
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import build_cnn
import numpy as np
from stable_baselines3.common.preprocessing import preprocess_obs
import torch
import torch as th
from torch import nn

logger = logging.getLogger(__name__)


from reward_preprocessing import utils
from reward_preprocessing.env import maze, mountain_car  # noqa: F401


class ProcgenCnnRegressionRewardNet(RewardNet):
    """Rewardnet using a CNN for learning reward using supervised regression on obs, rew
    pairs."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space=observation_space, action_space=action_space)

        # TODO: Not sure if Cnn (from this module) or build_cnn is better here. The
        # former gives us more freedom in the architecture.
        self.cnn_regressor = build_cnn(
            in_channels=3,
            hid_channels=[32, 64],
            out_size=1,
        )

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """
        Args:
            state: Tensor of shape (batch_size, height, width, channels)
            action: Tensor of shape (batch_size, action_size)
            next_state: Tensor of shape (batch_size, state_size)
            done: Tensor of shape (batch_size,)
        Returns:
            Tensor of shape (batch_size,)
        """
        # TODO: We always assume shape (batch_size, height, width, channels) for inputs,
        # do we actually want that or do we want to allow different shapes?
        # Performs preprocessing for images
        preprocessed_obs = preprocess_obs(
            next_state, self.observation_space, normalize_images=self.normalize_images
        )
        preprocessed_obs = cast(th.Tensor, preprocessed_obs)
        # Reshape to (batch_size, channels, height, width)
        if len(preprocessed_obs.shape) == 4:
            transposed = th.permute(preprocessed_obs, [0, 3, 1, 2])
        else:
            logging.warning(
                f"Encountered unexpected shape {preprocessed_obs.shape}. "
                "Skipping transpose."
            )
            transposed = preprocessed_obs
        return self.cnn_regressor(transposed)


class MazeRewardNet(RewardNet):
    def __init__(self, size: int, maze_name: str = "EmptyMaze", **kwargs):
        env = gym.make(f"reward_preprocessing/{maze_name}{size}-v0", **kwargs)
        self.rewards = env.rewards
        super().__init__(
            observation_space=env.observation_space, action_space=env.action_space
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        np_state = state.detach().cpu().numpy()
        np_next_state = next_state.detach().cpu().numpy()
        rewards = self.rewards[np_state, np_next_state]
        return torch.as_tensor(rewards, device=state.device)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.
        """
        state_th = torch.as_tensor(state, device=self.device, dtype=torch.long)
        action_th = torch.as_tensor(action, device=self.device)
        next_state_th = torch.as_tensor(
            next_state, device=self.device, dtype=torch.long
        )
        done_th = torch.as_tensor(done, device=self.device)

        assert state_th.shape == next_state_th.shape
        return state_th, action_th, next_state_th, done_th


class MountainCarRewardNet(RewardNet):
    def __init__(self, **kwargs):
        self.env = gym.make("imitation/MountainCar-v0", **kwargs).unwrapped
        super().__init__(self.env.observation_space, self.env.action_space)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        reward = (state[:, 0] > 0.5).float() - 1.0
        shaping = torch.tensor(
            [self.env._shaping(x, y) for x, y in zip(state, next_state)]
        )
        return reward + shaping

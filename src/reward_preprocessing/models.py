from typing import Tuple

import gym
import torch as th
from imitation.rewards.reward_nets import RewardNet
import numpy as np
import torch

from reward_preprocessing.env import maze, mountain_car  # noqa: F401

class CNNRegressionRewardNet(RewardNet):
    """CNN for learning reward using supervised regression from trajectories."""

    def __int__(self):

    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor,
                done: th.Tensor) -> th.Tensor:
        """
        Args:
            state: Tensor of shape (batch_size, state_size)
            action: Tensor of shape (batch_size, action_size)
            next_state: Tensor of shape (batch_size, state_size)
            done: Tensor of shape (batch_size,)
        Returns:
            Tensor of shape (batch_size,)
        """



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

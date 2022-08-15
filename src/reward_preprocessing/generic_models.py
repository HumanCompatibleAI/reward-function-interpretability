from imitation.rewards.reward_nets import RewardNet
import torch as th
from torch import nn


class CNN(RewardNet):
    """CNN for learning reward using supervised regression from trajectories."""

    def __int__(self):
        self.convs = nn.Sequential

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
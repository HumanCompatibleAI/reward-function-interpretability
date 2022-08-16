from typing import Tuple, List

import gym
import torch as th
from imitation.rewards.reward_nets import RewardNet
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from torch import nn

from reward_preprocessing import utils
from reward_preprocessing.env import maze, mountain_car  # noqa: F401

class CNNRegressionRewardNet(RewardNet):
    """CNN for learning reward using supervised regression from trajectories."""

    def __int__(self):

"


class SB3CnnObsRewardNet(RewardNet):
    """"""

    def __init__(self, env: gym.Env, device):
        super().__init__(observation_space=env.observation_space, action_space=env.action_space)
        self.features_extractor = PPO('CnnPolicy', env).policy.features_extractor
        features_dim = self.features_extractor.features_dim
        self.reward_net = nn.Linear(features_dim, 1).to(device)



    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor,
                done: th.Tensor) -> th.Tensor:
        """CNN reward net for image-based envs. Uses SB3's default CNN feature
        extractor architecture. Uses only the last frame of the observation to learn the reward.

        Args:
            state: Tensor of shape (batch_size, state_size)
            action: Tensor of shape (batch_size, action_size)
            next_state: Tensor of shape (batch_size, state_size)
            done: Tensor of shape (batch_size,)
        Returns:
            Tensor of shape (batch_size,)
        """
        last_frame = state[:, -1]
        self.forward(last_frame)

        # obs_transposed = VecTransposeImage.transpose_image(observation)
        latent, _, _ = self.ac_model._get_latent(
            th.tensor(obs_transposed).to(self.device))
        return self.reward_net(latent)



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


class CNN(nn.Module):
    """General CNN for learning reward using supervised regression from trajectories."""

    def __init__(self, input_size: int, in_channels: int, channels: List[int]):
        """
        Args:
            input_size:  The (scalar) size of the input image the conv net will be
            trained on
        :param input_size:
        :param in_channels: Number of channels of the input images
        :param channels: A list of the channel sizes for each layer
        """
        super().__init__()

        current_size = input_size
        # Base model
        self.model = nn.Sequential()

        # Create successive convolutional layers, with the in_channels coming from the
        # previous layer and the out_channels coming from the channels arguments
        previous = in_channels
        for i, out in enumerate(channels):
            new_module = nn.Conv2d(in_channels=previous, out_channels=out, kernel_size=(3, 3), stride=stride)
            self.model.add_module(name='conv' + str(i), module=new_module)
            self.model.add_module(name='relu' + str(i), module=nn.ReLU())
            previous = out
            # Update the input size after the current layer
            current_size = utils.calc_conv_out_size(current_size, 3, 0, stride)
        # For now only one pooling layer
        self.model.add_module('maxpool', nn.MaxPool2d(kernel_size=(2, 2)))
        # Size of the data can be calculated in the same way as with conv layers
        current_size = utils.conv_out_dim(current_size, 2, 0, 2)
        self.model.add_module('flatten', nn.Flatten())
        # Input dimensions * last number of channels is the number of params
        # in the flattened layer
        number_params = int(current_size * current_size * previous)
        # Single linear layer for this classifier
        self.model.add_module('linear', nn.Linear(number_params, num_classes))

    def forward(self, x):
        return self.model(x)

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

"""Excluded from LICENSE,
source: https://github.com/joonleesky/train-procgen-pytorch/
minor modifications, to make it an SB3 features extractor
"""
from typing import Tuple

import gym
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        **kwargs,
    ):
        super(ImpalaModel, self).__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=features_dim)

        self.output_dim = features_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class ImpalaGMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        """
        See stable_baselines3.common.policies.ActorCriticPolicy
        """
        super().__init__(*args, **kwargs)
        self.action_dim = self.action_space.n
        self.embedder = ImpalaModel(observation_space=self.observation_space)
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(
            nn.Linear(self.embedder.output_dim, self.action_dim), gain=0.01
        )
        self.fc_value = orthogonal_init(
            nn.Linear(self.embedder.output_dim, 1), gain=1.0
        )

        # self.recurrent = recurrent
        # if self.recurrent:
        #     self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def is_recurrent(self):
        return self.recurrent

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space,
                                          normalize_images=self.normalize_images)

        hidden = self.embedder(preprocessed_obs)
        # if self.recurrent:
        #     hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        # Below here I had to make changes to make it compatible with SB3
        # p = Categorical(logits=log_probs)  # From the original code
        distro = CategoricalDistribution(self.action_dim)
        distro = distro.proba_distribution(action_logits=log_probs)
        actions = distro.get_actions(deterministic=deterministic)
        v = self.fc_value(hidden).reshape(-1)
        log_prob = distro.log_prob(actions)
        return actions, v, log_prob

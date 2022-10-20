# Utilities for training a generative model on imitation rollouts.
# Ideally, you should only need to export rollouts_to_dataloader.

from pathlib import Path

from imitation.rewards.reward_nets import RewardNet
import torch as th
import torch.nn as nn
import vegans.utils

# TODO: add type annotations
from reward_preprocessing.common.utils import tensor_to_transition


class RewardGeneratorCombo(nn.Module):
    """Composition of a generative model and a RewardNet.

    Assumes that the RewardNet normalizes observations to [0,1].
    """

    def __init__(self, reward_net: RewardNet, generator: nn.Module):
        super().__init__()
        self.reward_net = reward_net
        self.generator = generator

    def forward(latent_vec):
        transition_tensor = generator(latent_vec)
        obs, action_vec, next_obs = tensor_to_transition(latent_vec)
        done = th.zeros(action_vec.shape)
        return reward_net.forward(obs, action_vec, next_obs, done)


def save_loss_plots(losses, save_dir):
    """Save plots of generator/adversary losses over training."""
    fig, _ = vegans.utils.plot_losses(losses, show=False)
    fig.savefig(Path(save_dir) / 'loss_fig.png')

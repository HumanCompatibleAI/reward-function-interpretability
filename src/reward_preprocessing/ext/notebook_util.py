"""Utils only used by the notebooks."""
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from reward_preprocessing.ext.impala import (
    ImpalaBlock,
    orthogonal_init,
    xavier_uniform_init,
)


# Same as ImpalaPolicy without the modifications to make it compatible with imitation.
class CategoricalPolicyGM(nn.Module):
    def __init__(
        self,
        embedder,
        # recurrent,
        action_size,
    ):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super().__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(
            nn.Linear(self.embedder.output_dim, action_size), gain=0.01
        )
        self.fc_value = orthogonal_init(
            nn.Linear(self.embedder.output_dim, 1), gain=1.0
        )

        # self.recurrent = recurrent
        # if self.recurrent:
        #     self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    # def is_recurrent(self):
    #     return self.recurrent

    def forward(self, x):  # , hx, masks):
        hidden = self.embedder(x)
        # if self.recurrent:
        #     hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v


scale = 1


# Used for the above
class ImpalaModel(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)
        self.relu_after_convs = nn.ReLU()

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu_after_convs(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def longest_common_prefix(l):
    l = set([s[: min(map(len, l))] for s in l])
    while len(l) > 1:
        l = set([s[:-1] for s in l])
    return list(l)[0]


def longest_common_suffix(l):
    l = set([s[-min(map(len, l)) :] for s in l])
    while len(l) > 1:
        l = set([s[1:] for s in l])
    return list(l)[0]


def get_abbreviator(names):
    """Small utility for abbreviating a list of names."""
    if len(names) <= 1:
        return slice(None, None)
    prefix = longest_common_prefix(names)
    prefix = prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else ""
    suffix = longest_common_suffix(names)
    suffix = "/" + suffix.split("/", 1)[-1] if "/" in suffix else ""
    return slice(len(prefix), None if len(suffix) == 0 else -len(suffix))

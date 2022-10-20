from imitation.rewards.reward_nets import RewardNetWrapper
import torch as th
import torch.nn as nn


class UncurryRewNet(RewardNetWrapper):
    """Wrapper to uncurry a reward net."""

    # We brake the signature of the constructor here because some external libraries
    # expect PyTorch models to only have one argument for forward(). In that case
    # this wrapper can be used to uncurry the reward net into a net that takes
    # exactly one tuple as argument
    def forward(self, tup: tuple) -> th.Tensor:
        kwargs = {
            "state": tup[0],
            "action": tup[1],
            "next_state": tup[2],
            "done": tup[3],
        }
        return self.base(**kwargs)


class NextStateOnlyModel(nn.Module):
    """Wrapper a reward net torch module. Wrapped model has single input in forward()
    for next_state, as opposed to 4 args."""

    def __init__(self, rew_net: nn.Module):
        super().__init__()
        self.rew_net = rew_net

    def forward(self, next_state: th.Tensor) -> th.Tensor:
        return self.rew_net(state=None, action=None, next_state=next_state, done=None)


class ChannelsFirstToChannelsLast(nn.Module):
    """Input: [bs, channels, height, width],
    passed on as: [bs, height, width, channels]."""

    def __init__(self, rew_net: nn.Module):
        super().__init__()
        self.rew_net = rew_net

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Input: [bs, channels, height, width]
        x = x.permute(0, 2, 3, 1)
        # Output: [bs, height, width, channels]
        # THis is what my reward nets expect
        return self.rew_net(x)


class FourDimOutput(nn.Module):
    """When output is 1D, for e.g. reward net, reshape so output is 4D for
    bs, height, width, channels."""

    def __init__(self, rew_net: nn.Module):
        super().__init__()
        self.rew_net = rew_net
        self.reshaped_out = ReshapeLayer()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.rew_net(x)
        out = self.reshaped_out(out)
        return out


class Repeat3Dim(nn.Module):
    """Used for debugging."""

    def __init__(self, net: nn.Module, target_shape: tuple):
        super().__init__()
        self.net = net
        self.target_shape = target_shape

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Input: [bs, 1, size, 1]
        new = th.zeros([x.shape[0]] + list(self.target_shape))
        minimum = min(x.shape[2], self.target_shape[2])
        new[:, :1, :minimum, :1] = x[:, :, :minimum, :]
        return self.net(new)


class ReshapeLayer(nn.Module):
    """Torch module that reshapes 1D to 4D."""

    def forward(self, x):

        return x.reshape((x.shape[0], 1, 1, 1))

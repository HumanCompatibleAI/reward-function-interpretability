"""Extend imitation's serialization by adding an additional reward net that can be
deserialized to the registry. This works, but we ended up using different
deserialization. We might end up never using this, in that case this can be removed."""
from imitation.rewards.serialize import ValidateRewardFn, _make_functional
from imitation.rewards.serialize import load_reward as imitation_load_reward
from imitation.rewards.serialize import reward_registry
import torch as th


def _loader_helper(path, _, **kwargs):
    # AFAICT by default imitation will load saved reward nets and then torch will move
    # them to the same devices they are saved on. This fails if e.g. GPU is not
    # available in the environment that loads. The following circumvents this problem
    # by using torch.load's map_location argument.
    if th.cuda.is_available():
        rew_net = th.load(str(path))  # Load from same device as saved
    else:  # CUDA not available
        rew_net = th.load(str(path), map_location=th.device("cpu"))  # Force CPU
    return ValidateRewardFn(_make_functional(rew_net))


# Register our custom reward so that imitation internals can e.g. load the specified
# type. We register in a very similar way to imitation.reward.serialize
reward_registry.register(
    key="SupervisedRewardNet",
    # Validate the shape returned by the reward net
    value=_loader_helper,
)


# Wrap the original imitation function so that we can use our custom reward by
# registering it above.
load_reward = imitation_load_reward

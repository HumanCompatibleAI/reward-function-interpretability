from imitation.rewards.serialize import ValidateRewardFn, _make_functional
from imitation.rewards.serialize import load_reward as imitation_load_reward
from imitation.rewards.serialize import reward_registry
import torch as th

# Register our custom reward so that imitation internals can e.g. load the specified
# type. We register in a very similar way to imitation.reward.serialize
reward_registry.register(
    key="SupervisedRewardNet",
    # Validate the shape returned by the reward net
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(  # Turns the loaded reward net into a reward function.
            th.load(str(path)),
        ),
    ),
)


# Wrap the original imitation function so that we can use our custom reward by
# registering it above.
load_reward = imitation_load_reward

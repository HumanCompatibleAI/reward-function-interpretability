import logging
from typing import Tuple

import gym
from imitation.rewards.reward_nets import CnnRewardNet, RewardNet
import numpy as np
import torch as th

from reward_preprocessing.env import maze, mountain_car  # noqa: F401

logger = logging.getLogger(__name__)


class CnnRewardNetWorkaround(CnnRewardNet):
    """Identical to CnnRewardNet, except that it fixes imitation issue #644 by
    removing normalize_input_layer from the kwargs.
    TODO: Reconsider this once the underlying issue is fixed.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        hwc_format: bool = True,
        **kwargs,
    ):
        normalize = kwargs.pop("normalize_input_layer", None)
        if normalize is not None:
            logger.warning(
                f"normalize_input_layer={normalize} was provided, will be ignored. See "
                "imitation issue #64.4"
            )

        super().__init__(
            observation_space,
            action_space,
            use_state,
            use_action,
            use_next_state,
            use_done,
            hwc_format,
            **kwargs,
        )


class MazeRewardNet(RewardNet):
    def __init__(self, size: int, maze_name: str = "EmptyMaze", **kwargs):
        env = gym.make(f"reward_preprocessing/{maze_name}{size}-v0", **kwargs)
        self.rewards = env.rewards
        super().__init__(
            observation_space=env.observation_space, action_space=env.action_space
        )

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ):
        np_state = state.detach().cpu().numpy()
        np_next_state = next_state.detach().cpu().numpy()
        rewards = self.rewards[np_state, np_next_state]
        return th.as_tensor(rewards, device=state.device)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.
        """
        state_th = th.as_tensor(state, device=self.device, dtype=th.long)
        action_th = th.as_tensor(action, device=self.device)
        next_state_th = th.as_tensor(next_state, device=self.device, dtype=th.long)
        done_th = th.as_tensor(done, device=self.device)

        assert state_th.shape == next_state_th.shape
        return state_th, action_th, next_state_th, done_th


class MountainCarRewardNet(RewardNet):
    def __init__(self, **kwargs):
        self.env = gym.make("imitation/MountainCar-v0", **kwargs).unwrapped
        super().__init__(self.env.observation_space, self.env.action_space)

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        reward = (state[:, 0] > 0.5).float() - 1.0
        shaping = th.tensor(
            [self.env._shaping(x, y) for x, y in zip(state, next_state)]
        )
        return reward + shaping

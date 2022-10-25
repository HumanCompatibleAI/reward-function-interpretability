import logging
from typing import Tuple, Optional

import gym
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import build_cnn
import numpy as np
from stable_baselines3.common.preprocessing import preprocess_obs
import torch as th

from reward_preprocessing.env import maze, mountain_car  # noqa: F401

logger = logging.getLogger(__name__)

def _make_concat_inputs(action_space, regressor_input):
    """Produces a function that takes the relevant inputs to a reward net, that being
    state, action, next_state, as a tensor and concatenates them according to which
    are relevant for the regressor_input.

    For use at the beginning of the forward pass of a reward net.

    Args:
        action_space: The action space of the environment.
        regressor_input: The input to the regressor, as passed to SupervisedRewardNet.
            This function will not double-check validity of regressor_input.
    """

    num_acts = action_space.shape[0]
    use_state = "state" in regressor_input
    use_action = "action" in regressor_input
    use_next_state = "next_state" in regressor_input

    def concat_inputs(
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
    ) -> th.Tensor:

        transp_obs = np.transpose(obs, (2, 0, 1))
        obs_height = transp_obs.shape[1]
        obs_width = transp_obs.shape[2]
        act_one_hot = np.zeros(num_acts)
        act_one_hot[act] = 1
        act_one_hot = act_one_hot[:, None, None]
        boosted_act = np.broadcast_to(act_one_hot, (num_acts, obs_height, obs_width))
        transp_next_obs = np.transpose(next_obs, (2, 0, 1))
        assert transp_next_obs.shape[1] == obs_height
        assert transp_next_obs.shape[2] == obs_width
        tensor_transition = np.concatenate(
            [transp_obs, boosted_act, transp_next_obs],
            axis=0,
        )
        return tensor_transition

    return transition_to_tensor


class ProcgenCnnRegressionRewardNet(RewardNet):
    """RewardNet using a CNN for learning reward using supervised regression on obs, rew
    pairs."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        regressor_input: Optional[tuple] = None,
        cnn_kwargs: Optional[dict] = None,
    ):
        """
        Build a CNN-based reward network.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            regressor_input:
                Tuple of strings indicating which inputs the neural net
                should take. Any combination of ("state", "action", "next_state").
                If None, defaults to ("state", "action", "next_state").
            cnn_kwargs:
                Keyword arguments to pass to build_cnn. See
                imitation.util.networks.build_cnn for details.
                Do not pass in_channels, these will be determined from
                observation_space, action_space, and regressor_input.
                Do not pass out_size, this will always be 1.
                hid_channels is required.
                If None defaults to default arguments for build_cnn with
                hid_channels=[32, 64].
        """
        super().__init__(observation_space=observation_space, action_space=action_space)

        if regressor_input is None:
            regressor_input = ("state", "action", "next_state")
        elif len(regressor_input) == 0:
            raise ValueError(
                "regressor_input must be non-empty. Use None for defaults."
            )
        elif len(regressor_input) >= 4:
            raise ValueError(
                "regressor_input must be any combination of "
                "('state', 'action', 'next_state')."
            )
        if cnn_kwargs is None:
            cnn_kwargs = dict(hid_channels=[32, 64])
        elif "in_channels" in cnn_kwargs or "out_size" in cnn_kwargs:
            raise ValueError(
                "cnn_kwargs must not contain 'in_channels' or 'out_size'. These will "
                "be set automatically."
            )
        if "hid_channels" not in cnn_kwargs:
            raise ValueError("cnn_kwargs must contain 'hid_channels'.")

        self.regressor_input = regressor_input
        self.cnn_kwargs = cnn_kwargs

        # Determine number of input channels.
        in_channels = 0
        if "state" in regressor_input:
            in_channels += observation_space.shape[-1]
        if "action" in regressor_input:
            in_channels += action_space.shape[0]
        if "next_state" in regressor_input:
            in_channels += observation_space.shape[-1]

        self.cnn_regressor = build_cnn(
            in_channels=in_channels,
            out_size=1,
            **cnn_kwargs,
        )

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """
        Args:
            state: Tensor of shape (batch_size, height, width, channels)
            action: Tensor of shape (batch_size, action_size)
            next_state: Tensor of shape (batch_size, state_size)
            done: Tensor of shape (batch_size,)
        Returns:
            Tensor of shape (batch_size,)
        """
        # TODO: We always assume shape (batch_size, height, width, channels) for inputs,
        # do we actually want that or do we want to allow different shapes?
        # Performs preprocessing for images
        preprocessed_obs = preprocess_obs(
            next_state, self.observation_space, normalize_images=self.normalize_images
        )
        assert isinstance(preprocessed_obs, th.Tensor)
        # Reshape from (batch_size [0], height [1], width [2], channels [3])
        # to (batch_size [0], channels [3], height [1], width [2])
        if len(preprocessed_obs.shape) == 4:
            transposed = th.permute(preprocessed_obs, [0, 3, 1, 2])
        else:
            logging.warning(
                f"Encountered unexpected shape {preprocessed_obs.shape}. "
                "Skipping transpose."
            )
            transposed = preprocessed_obs
        batch_size = transposed.shape[0]

        # Reshape into shape expected by imitation (see RewardNet predict_th())
        out = self.cnn_regressor(transposed).reshape((batch_size,))
        return out


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

"""Code to register procgen environments to train reward funcs on."""

import re
from typing import Iterable

import gym
from seals.util import AutoResetWrapper, get_gym_max_episode_steps


def supported_procgen_env(gym_spec: gym.envs.registration.EnvSpec) -> bool:
    starts_with_procgen = gym_spec.id.startswith("procgen-")
    three_parts = len(re.split("-|_", gym_spec.id)) == 3
    return starts_with_procgen and three_parts


def make_auto_reset_procgen(procgen_env_id: str, **make_env_kwargs) -> gym.Env:
    """Make procgen with auto reset. Final observation is not fixed.

    That means the final observation will be a duplicate of the second to last."""
    env = AutoResetWrapper(gym.make(procgen_env_id, **make_env_kwargs))
    return env


def make_fin_obs_auto_reset_procgen(procgen_env_id: str, **make_env_kwargs) -> gym.Env:
    """Make procgen with auto reset and fixed final observation."""
    # The order of the wrappers matters here. Final obs wrapper must be applied first,
    # then auto reset wrapper. This is because the final obs wrapper depends on the
    # done signal, on order to fix the final observation of an episode. The auto reset
    # wrapper will reset the done signal to False for the original episode end.
    env = AutoResetWrapper(
        ProcgenFinalObsWrapper(gym.make(procgen_env_id, **make_env_kwargs))
    )
    return env


def make_fin_obs_procgen(procgen_env_id: str, **make_env_kwargs) -> gym.Env:
    """Make procgen with fixed final observation."""
    env = ProcgenFinalObsWrapper(gym.make(procgen_env_id, **make_env_kwargs))
    return env


def local_name_autoreset(gym_spec: gym.envs.registration.EnvSpec) -> str:
    split_str = gym_spec.id.split("-")
    version = split_str[-1]
    split_str[-1] = "autoreset"
    return "-".join(split_str + [version])


def local_name_fin_obs(gym_spec: gym.envs.registration.EnvSpec) -> str:
    split_str = gym_spec.id.split("-")
    version = split_str[-1]
    split_str[-1] = "final-obs"
    return "-".join(split_str + [version])


def local_name_fin_obs_autoreset(gym_spec: gym.envs.registration.EnvSpec) -> str:
    split_str = gym_spec.id.split("-")
    version = split_str[-1]
    split_str[-1] = "final-obs-autoreset"
    return "-".join(split_str + [version])


def register_procgen_envs(
    gym_procgen_env_specs: Iterable[gym.envs.registration.EnvSpec],
) -> None:

    to_register = [
        # Auto reset with original final observation behavior.
        (local_name_autoreset, "reward_preprocessing.procgen:make_auto_reset_procgen"),
        # Variable-length procgen with fixed final observation.
        (local_name_fin_obs, "reward_preprocessing.procgen:make_fin_obs_procgen"),
        # Fixed-length procgen with fixed final observation.
        (
            local_name_fin_obs_autoreset,
            "reward_preprocessing.procgen:make_fin_obs_auto_reset_procgen",
        ),
    ]
    for (local_name_fn, entry_point) in to_register:
        for gym_spec in gym_procgen_env_specs:
            gym.envs.registration.register(
                id=local_name_fn(gym_spec),
                entry_point=entry_point,
                max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
                kwargs=dict(procgen_env_id=gym_spec.id),
            )


class ProcgenFinalObsWrapper(gym.Wrapper):
    """Returns the final observation of gym3 procgen environment, correcting for the
    fact that Procgen gym environments return the second-to-last observation again
    instead of the final observation.
    
    Only works correctly when the 'done' signal coincides with the end of an episode
    (which is not the case when using e.g. the seals AutoResetWrapper).
    Requires the use of the PavelCz/procgenAISC fork, which adds the 'final_obs' value.

    Since procgen builds on gym3, it always resets the environment after a terminal
    state. The final 'obs' returned when done==True will be the obs that was already
    returned in the previous step. In our fork of procgen, we save the true last
    observation of the terminated episode in the info dict. This wrapper extracts that
    obs and returns it.
    """

    def step(self, action):
        """When done=True, returns the final_obs from the dict."""
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = info["final_obs"]
        return obs, rew, done, info

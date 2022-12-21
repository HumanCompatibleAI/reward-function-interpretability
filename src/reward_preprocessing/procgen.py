"""Code to register procgen environments to train reward funcs on."""

import re
from typing import Iterable

import gym
from seals.util import AutoResetWrapper, get_gym_max_episode_steps

from reward_preprocessing.common.utils import ProcgenFinalObsWrapper


def supported_procgen_env(gym_spec: gym.envs.registration.EnvSpec) -> bool:
    starts_with_procgen = gym_spec.id.startswith("procgen-")
    three_parts = len(re.split("-|_", gym_spec.id)) == 3
    return starts_with_procgen and three_parts


def make_auto_reset_procgen(procgen_env_id: str, **make_env_kwargs) -> gym.Env:
    env = AutoResetWrapper(gym.make(procgen_env_id, **make_env_kwargs))
    return env


def make_fin_obs_procgen(procgen_env_id: str, **make_env_kwargs) -> gym.Env:
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


def register_procgen_envs(
    gym_procgen_env_specs: Iterable[gym.envs.registration.EnvSpec],
) -> None:

    for gym_spec in gym_procgen_env_specs:
        gym.register(
            id=local_name(gym_spec),
            entry_point="reward_preprocessing.procgen:make_auto_reset_procgen",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(procgen_env_id=gym_spec.id),
        )

    # There are no envs that have both autoreset and final obs wrappers.
    # fin-obs would only affect the terminal_observation in the info dict, if it were
    # to be wrapped by an AutoResetWrapper. Since, at the moment, we don't use the
    # terminal_observation in the info dict, there is no point to combining them.
    for gym_spec in gym_procgen_env_specs:
        gym.register(
            id=local_name_fin_obs(gym_spec),
            entry_point="reward_preprocessing.procgen:make_fin_obs_procgen",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(procgen_env_id=gym_spec.id),
        )

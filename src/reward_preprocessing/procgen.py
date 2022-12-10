"""Code to register procgen environments to train reward funcs on."""

from typing import Iterable

import gym
from seals.util import AutoResetWrapper, get_gym_max_episode_steps


def supported_procgen_env(gym_spec: gym.envs.registration.EnvSpec) -> bool:
    return gym_spec.id.startswith("procgen:procgen-")


def make_auto_reset_procgen(procgen_env_id: str) -> gym.Env:
    env = AutoResetWrapper(gym.make(procgen_env_id))
    return env


def local_name(gym_spec: gym.envs.registration.EnvSpec) -> str:
    return gym_spec.id + "-autoreset"


def register_procgen_envs(
    gym_procgen_env_specs: Iterable[gym.envs.registration.EnvSpec],
) -> None:

    for gym_spec in gym_procgen_env_specs:
        gym.register(
            id=local_name(gym_spec.id),
            entry_point="reward_preprocessing.procgen:make-aute_reset_procgen",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(procgen_env_id=gym_spec.id),
        )
        print(local_name(gym_spec.id))

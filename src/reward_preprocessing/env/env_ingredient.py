from typing import Any, Iterable, Mapping

import gym
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reward_preprocessing.utils import instantiate

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    name = "EmptyMaze-v0"
    options = {}
    stats_path = None
    wrappers = []
    n_envs = 1
    normalize = False
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def empty_maze():
    # this is currently the default anyway, but it's needed to
    # make the wrapper script work
    name = "EmptyMaze-v0"
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def mountain_car():
    name = "MountainCar-v0"
    stats_path = "results/stats/MountainCar-v0.pkl"
    normalize = True
    _ = locals()  # make flake8 happy
    del _


@env_ingredient.named_config
def half_cheetah():
    name = "HalfCheetah-v2"
    wrappers = ["sb3_contrib.common.wrappers.TimeFeatureWrapper"]
    n_envs = 16
    normalize = True
    _ = locals()  # make flake8 happy
    del _


def wrap_env(env: gym.Env, wrappers: Iterable[str]) -> gym.Env:
    """Wrap a gym env in multiple wrappers, given by their class name.

    `wrappers` must be an Iterable of complete paths to the
    class, including the package and module.
    """
    for wrapper_name in wrappers:
        # wrapper_name is a complete class, like
        # sb3_contrib.common.wrappers.TimeFeatureWrapper.
        # We want to split that into module and class name:
        module_name, _, cls_name = wrapper_name.rpartition(".")
        env = instantiate(module_name, cls_name, env=env)
    return env


@env_ingredient.capture
def create_env(
    name: str,
    _seed: int,
    options: Mapping[str, Any],
    stats_path: str,
    n_envs: int,
    normalize: bool,
    wrappers: Iterable[str],
):
    env = DummyVecEnv([lambda: wrap_env(gym.make(name, **options), wrappers)] * n_envs)
    env.seed(_seed)
    # the action space uses a distinct random seed from the environment
    # itself, which is important if we use randomly sampled actions
    env.action_space.np_random.seed(_seed)

    if normalize:
        if stats_path:
            env = VecNormalize.load(stats_path, env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env

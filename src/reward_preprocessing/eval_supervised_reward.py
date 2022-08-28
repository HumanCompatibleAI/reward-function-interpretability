import logging
import os.path as osp
from typing import Mapping, Optional

from imitation.data import rollout
from imitation.data.rollout import AnyPolicy
from imitation.policies import serialize
from imitation.rewards import reward_function
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import (
    ValidateRewardFn,
    _make_functional,
    load_reward,
    reward_registry,
)
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.vec_env import VecEnv
import torch as th

from reward_preprocessing.env import create_env, env_ingredient

eval_supervised_ex = Experiment("eval_supervised", ingredients=[env_ingredient])

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


def _eval_policy(
    venv: VecEnv,
    num_episodes: int = 100,
    policy: AnyPolicy = None,
    reward_fn: Optional[reward_function.RewardFn] = None,
) -> Mapping[str, float]:
    """Roll out a policy in an environment with optional reward function.

    Args:
        venv: The environment to roll out in.
        num_episodes: Number of episodes to roll out.
        policy: Optional policy to roll out. If None sample randomly.
        reward_fn: Optional reward function to evaluate the policy with.
    Returns:
         Return value of `imitation.util.rollout.rollout_stats()`.
    """
    try:
        if reward_fn is not None:
            venv = RewardVecEnvWrapper(venv, reward_fn)
        sample_until = rollout.make_sample_until(
            min_episodes=num_episodes, min_timesteps=None
        )
        trajectories = rollout.generate_trajectories(policy, venv, sample_until)

        return rollout.rollout_stats(trajectories)
    finally:
        venv.close()


@eval_supervised_ex.config
def defaults():
    # Number of episodes for each evaluation
    eval_n_episodes = 50
    # Type of expert policy to use
    policy_type = "ppo"
    # Path to the expert policy
    policy_path = None
    # Path to the learned supervised reward net
    reward_path = None

    locals()  # quieten flake8


@eval_supervised_ex.main
def eval_reward(
    eval_n_episodes: int,
    policy_type: str,
    policy_path: Optional[str],
    reward_path: Optional[str],
):
    """Sanity check a learned supervised reward net. Evaluate 4 things:
    - Random policy on env reward
    - Random policy on learned reward function
    - Expert policy on env reward
    - Expert policy on learned reward function
    """
    venv = create_env()

    rew_fn = load_reward("SupervisedRewardNet", reward_path, venv)

    policy = None
    if policy_type is not None:
        policy = serialize.load_policy(policy_type, policy_path, venv)

    logging.info(f"Evaluating random policy with env reward as sanity check.")
    stats = _eval_policy(venv, num_episodes=eval_n_episodes)
    logging.info(stats)

    logging.info(f"Evaluating random policy on learned reward.")
    stats = _eval_policy(venv, num_episodes=eval_n_episodes, reward_fn=rew_fn)
    logging.info(stats)

    logging.info(f"Evaluating expert policy on env reward.")
    stats = _eval_policy(venv, num_episodes=eval_n_episodes, policy=policy)
    logging.info(stats)

    logging.info(f"Evaluating expert policy on learned reward.")
    stats = _eval_policy(
        venv, num_episodes=eval_n_episodes, policy=policy, reward_fn=rew_fn
    )
    logging.info(stats)


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "eval_supervised"))
    eval_supervised_ex.observers.append(observer)
    eval_supervised_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

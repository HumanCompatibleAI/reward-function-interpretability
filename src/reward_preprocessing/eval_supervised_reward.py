import logging
import os.path as osp
from typing import Optional

from imitation.rewards.serialize import (
    ValidateRewardFn,
    _make_functional,
    reward_registry,
)
from imitation.scripts import eval_policy

# I need to import common because eval_policy uses  common.make_log_dir(). If I
# don't import it, eval_policy will compalin that it is missing ['log_dir', 'log_level']
# and I don't have any way to pass these (AFAICT). So instead I import it here and
# make common part of my experiment.
from imitation.scripts.common import common
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.env import env_ingredient

eval_supervised_ex = Experiment(
    "eval_supervised", ingredients=[env_ingredient, common.common_ingredient]
)

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
    eval_n_episodes: int,
    policy_type: Optional[str],
    reward_type: Optional[str],
    policy_path: Optional[str] = None,
    reward_path: Optional[str] = None,
):
    """Wrapper to make calling eval_policy more convenient."""

    return eval_policy.eval_policy(
        eval_n_episodes=eval_n_episodes,
        policy_type=policy_type,
        policy_path=policy_path,
        reward_type=reward_type,
        reward_path=reward_path,
        eval_n_timesteps=None,
        render=False,
        render_fps=60,
        videos=False,
        video_kwargs={},
    )


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


@eval_supervised_ex.main
def main(
    eval_n_episodes: int,
    policy_type: str,
    policy_path: Optional[str],
    reward_path: Optional[str],
):
    if th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # env = gym.make(f"procgen:procgen-{env_name}-v0", render_mode="rgb_array")
    # env_max_steps = MaxEpisodeStepsWrapper(env, max_episode_steps=100)
    # venv = DummyVecEnv([lambda: env])
    # venv_max_steps = DummyVecEnv([lambda: env_max_steps])

    # reward_net = CoinReward()

    # print("Eval before training")
    # rollout_stats_before_training = eval_policy(venv_max_steps)
    # print(rollout_stats_before_training)

    logging.info(f"Evaluating random policy with env reward as sanity check.")
    stats = _eval_policy(
        eval_n_episodes=eval_n_episodes,
        policy_type=None,
        reward_type=None,
    )
    logging.info(stats)

    logging.info(f"Evaluating random policy on learned reward.")
    stats = _eval_policy(
        eval_n_episodes=eval_n_episodes,
        policy_type=None,
        reward_type="SupervisedRewardNet",
        reward_path=reward_path,
    )
    logging.info(stats)

    logging.info(f"Evaluating expert policy on env reward.")
    stats = _eval_policy(
        eval_n_episodes=eval_n_episodes,
        policy_type=policy_type,
        policy_path=policy_path,
        reward_type=None,
    )
    logging.info(stats)

    logging.info(f"Evaluating expert policy on learned reward.")
    stats = _eval_policy(
        eval_n_episodes=eval_n_episodes,
        policy_type=policy_type,
        policy_path=policy_path,
        reward_type="SupervisedRewardNet",
    )
    logging.info(stats)


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "eval_supervised"))
    eval_supervised_ex.observers.append(observer)
    eval_supervised_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

"""Thin wrapper around imitation's eval_policy script.

If you're not Daniel, you probably need to change these paths.
"""
from imitation.scripts.config.eval_policy import eval_policy_ex
from imitation.scripts.eval_policy import main_console

from reward_preprocessing.env.maze import use_config

use_config(eval_policy_ex)


@eval_policy_ex.named_config
def maze():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training_2023-02/maze/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 3e4
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "maze_rollouts_3e4_episodes_2023-02.npz"
    )
    common = dict(env_name="procgen:procgen-maze-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def bigfish():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training_2023-02/bigfish/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 1e5
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "bigfish_rollouts_1e5_episodes_2023-02.npz"
    )
    common = dict(env_name="procgen:procgen-bigfish-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def maze_10_eps():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training_2023-02/maze/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 10
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "maze_rollouts_10_episodes_2023-02.npz"
    )
    common = dict(env_name="procgen:procgen-maze-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def bigfish_20_eps():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training_2023-02/bigfish/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 20
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "bigfish_rollouts_20_episodes_2023-02.npz"
    )
    common = dict(env_name="procgen:procgen-bigfish-v0")
    locals()  # make flake8 happy


if __name__ == "__main__":
    main_console()

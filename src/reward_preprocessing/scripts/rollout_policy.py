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
    common = dict(env_name="procgen:procgen-maze-final-obs-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def heist():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training/heist/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 3e3
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "heist_rollouts_3e3_episodes_2023-05.npz"
    )
    common = dict(env_name="procgen:procgen-heist-final-obs-v0")
    locals()


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
    common = dict(env_name="procgen:procgen-bigfish-final-obs-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def coinrun():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training_2023-02/coinrun/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 3e3
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "coinrun_rollouts_3e3_episodes_2023-04.npz"
    )
    common = dict(env_name="procgen:procgen-coinrun-final-obs-v0")
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
    common = dict(env_name="procgen:procgen-maze-final-obs-v0")
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
    common = dict(env_name="procgen:procgen-bigfish-final-obs-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def coinrun_5_eps():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "output/train_rl/procgen:procgen-coinrun-v0/"
                + "20230322_161913_02c432/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 5
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "coinrun_rollouts_5_episodes_2023-04.npz"
    )
    common = dict(env_name="procgen:procgen-coinrun-final-obs-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def bigfish_5_eps():
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
        + "bigfish_rollouts_5_episodes_2023-04.npz"
    )
    common = dict(env_name="procgen:procgen-bigfish-final-obs-v0")
    locals()  # make flake8 happy


@eval_policy_ex.named_config
def explore_tiny_amount():
    explore_kwargs = dict(switch_prob=1.0, random_prob=0.01)
    locals()


@eval_policy_ex.named_config
def heist_100_eps():
    expert = dict(
        policy_type="ppo",
        loader_kwargs=dict(
            path=(
                "/home/daniel/reward-function-interpretability/"
                + "procgen_training/heist/policies/final/"
            ),
        ),
    )
    eval_n_episodes = 100
    rollout_save_path = (
        "/home/daniel/reward-function-interpretability/"
        + "heist_rollouts_100_episodes_2023-04.npz"
    )
    common = dict(env_name="procgen:procgen-heist-final-obs-v0")
    locals()


if __name__ == "__main__":
    main_console()

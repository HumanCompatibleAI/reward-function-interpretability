"""Thin wrapper around imitation's train_rl script to """
from imitation.scripts.config.train_rl import train_rl_ex
from imitation.scripts.train_rl import main_console

from reward_preprocessing.env.maze import use_config
from reward_preprocessing.ext.impala import ImpalaModel
import reward_preprocessing.policies.base

use_config(train_rl_ex)


@train_rl_ex.named_config
def coinrun_aisc_fixed_coin():
    """Coinrun with hparams from goal misgeneralization paper."""
    # Hparams inferred from https://github.com/jbkjr/train-procgen-pytorch/blob/master/hyperparams/procgen/config.yml
    # (hard-500)
    # This is the procgen version from procgenAISC, but the version that has the coin
    # always fixed at the end of the level, as in the original coinrun.
    train = dict(
        policy_cls=reward_preprocessing.policies.base.ImpalaPolicy,
        # policy_kwargs=dict(features_extractor_class=ImpalaModel),
    )
    policy_save_interval = 1_000_000
    common = dict(
        env_name="procgen:procgen-coinrun-v0",
        num_vec=256,  # Goal Misg paper uses 64 envs for each of 4 workers.
        env_make_kwargs=dict(num_levels=100_000, distribution_mode="hard"),
    )
    total_timesteps = int(200_000_000)
    rl = dict(
        batch_size=256,  # This corresponds to n_steps for rollouts.
        rl_kwargs=dict(
            n_epochs=3,
            ent_coef=0.01,
            learning_rate=0.0005,
            batch_size=8192,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=0.5,
            vf_coef=0.5,
            normalize_advantage=True,
        ),
    )
    # num_vec * n_steps / (mini)batch_size amounts to 8 minibatches as in the original
    # paper. They also did this in parallel for 4 workers but we don't do that here.
    locals()  # make flake8 happy


@train_rl_ex.named_config
def empty_maze_10():
    common = dict(env_name="reward_preprocessing/EmptyMaze10-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def empty_maze_4():
    common = dict(env_name="reward_preprocessing/EmptyMaze4-v0")
    total_timesteps = int(1e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def key_maze_10():
    common = dict(env_name="reward_preprocessing/KeyMaze10-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


@train_rl_ex.named_config
def key_maze_6():
    common = dict(env_name="reward_preprocessing/KeyMaze6-v0")
    total_timesteps = int(5e5)
    normalize = False
    locals()  # make flake8 happy


if __name__ == "__main__":  # pragma: no cover
    main_console()

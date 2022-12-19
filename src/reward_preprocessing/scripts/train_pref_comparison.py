"""Thin wrapper around imitation's train_preference_comparisons script."""
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.scripts.train_preference_comparisons import main_console

from reward_preprocessing.env.maze import use_config
import reward_preprocessing.policies.base

use_config(train_preference_comparisons_ex)


@train_preference_comparisons_ex.named_config
def coinrun():
    """Training with preference comparisons on coinrun."""
    fragment_length = 200
    total_comparisons = 100_000
    total_timesteps = 200_000_000
    # agent_path = "/home/daniel/reward-function-interpretability/
    # procgen/coinrun/policies/final/model.zip"  # move this to nas?
    train = dict(
        policy_cls=reward_preprocessing.policies.base.ImpalaPolicy,
    )
    common = dict(
        env_name="procgen:procgen-coinrun-autoreset-v0",
        num_vec=256,  # Goal Misg paper uses 64 envs for each of 4 workers.
        env_make_kwargs=dict(num_levels=100_000, distribution_mode="hard"),
    )
    rl = dict(
        batch_size=256 * 256,
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
    locals()  # make flake8 happy


@train_preference_comparisons_ex.named_config
def fast_procgen():  # Overrides some settings for fast setup for debugging purposes.
    rl = dict(batch_size=32, rl_kwargs=dict(batch_size=16))
    common = dict(num_vec=1)
    total_comparisons = 100
    fragment_length = 50
    total_timesteps = 1_000
    locals()  # make flake8 happy


if __name__ == "__main__":  # pragma: no cover
    main_console()

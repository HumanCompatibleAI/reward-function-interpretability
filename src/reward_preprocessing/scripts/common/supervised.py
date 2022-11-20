"""Common configuration elements for training supervised models."""

import logging
from typing import Any, Mapping, Optional, Sequence

from imitation.data import types
from imitation.rewards.reward_nets import RewardNet
from imitation.util import logger as imit_logger
import sacred
import torch as th

from reward_preprocessing.trainers.supervised_trainer import SupervisedTrainer

supervised_ingredient = sacred.Ingredient("supervised")
logger = logging.getLogger(__name__)


@supervised_ingredient.config
def config():
    epochs = 100  # Number of training epochs
    test_frac = 0.1  # Fraction of training data to use for testing
    test_freq = 64  # Frequency of running tests (in batches)
    batch_size = 32  # Batch size for training a supervised model
    num_loader_workers = 0  # Number of workers for data loading
    # Limit the total number of samples (train and test) to this number. Default of -1
    # to not limit the number of samples.
    limit_samples = -1

    # Apparently in sacred I need default values for parameters that I want to be able
    # to override. At least that's how I interpret this information:
    # https://github.com/IDSIA/sacred/issues/644

    # Keyword arguments for reward network
    net_kwargs = dict(
        use_state=True, use_action=True, use_next_state=True, hid_channels=(32, 64)
    )
    # Keyword arguments for Adam optimizer
    opt_kwargs = dict(lr=1e-3)

    debugging = dict(
        disable_dataset_shuffling=False,
        # Log samples from the dataset in wandb. True to show all samples. Also turns
        # the samples into a video. Set disable_dataset_shuffling to True to ensure
        # transitions in video are sequential. This will require installing further
        # packages for creating the video.
        show_samples=False,
        # If show_samples is False, show_samples_as_step has no effect.
        # If show_samples is True, setting show_samples_as_step to True will log all
        # samples (transitions) as a single panel in wandb. This panel will have a
        # slider to step through the samples.
        # Setting show_samples_as_step to False, there will be a separate panel in wandb
        # for every transition.
        show_samples_as_step=True,
    )

    locals()  # quieten flake8


@supervised_ingredient.config_hook
def config_hook(config, command_name, logger) -> dict:
    """Warn if network is set to `use_done`, since this setting will be overriden
    in train_regression."""
    del command_name
    res = {}
    if (
        "use_done" in config["supervised"]["net_kwargs"]
        and config["supervised"]["net_kwargs"]["use_done"]
    ):
        logger.warning(
            "Supervised training does not support setting use_done to "
            "True. We don't support networks that take in the done signal. "
            "This value will be ignored."
        )

    return res


@supervised_ingredient.capture
def make_trainer(
    expert_trajectories: Sequence[types.TrajectoryWithRew],
    model: RewardNet,
    custom_logger: Optional[imit_logger.HierarchicalLogger],
    test_frac: float,
    test_freq: int,
    batch_size: int,
    num_loader_workers: int,
    limit_samples: int,
    opt_kwargs: Optional[Mapping[str, Any]],
    debugging: Mapping,
) -> SupervisedTrainer:
    # MSE loss with mean reduction (the default)
    # Mean reduction means every batch affects model updates the same, regardless of
    # batch_size.
    loss_fn = th.nn.MSELoss()

    trainer = SupervisedTrainer(
        demonstrations=expert_trajectories,
        limit_samples=limit_samples,
        reward_net=model,
        batch_size=batch_size,
        test_frac=test_frac,
        test_freq=test_freq,
        num_loader_workers=num_loader_workers,
        loss_fn=loss_fn,
        opt_kwargs=opt_kwargs,
        custom_logger=custom_logger,
        allow_variable_horizon=True,
        debug_settings=debugging,
    )
    return trainer

"""Common configuration elements for training supervised models."""

import logging
import math
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
    test_freq = 256  # Frequency of running tests (in batches)
    batch_size = 128  # Batch size for training a supervised model
    num_loader_workers = 0  # Number of workers for data loading
    # Limit the total number of samples (train and test) to this number. Default of -1
    # to not limit the number of samples.
    limit_samples = -1
    # Only evaluate test loss on 4 batches when you're in the middle of a train epoch.
    # Set to None to evaluate on the whole test set.
    test_subset_within_epoch = 4
    # train classification for whether reward is 0 or not, rather than regression.
    classify = False
    # use adversarial training. below are configs to be set if adversarial is set to
    # True. for details, see documentation of SupervisedTrainer in
    # trainers/supervised_trainer.py
    adversarial = False
    start_epoch = None
    nonsense_reward = None
    vis_frac_per_epoch = None
    gradient_clip_percentile = None
    # Retain this fraction of zero-reward transitions and filter out the rest to
    # manually re-weight the dataset. Default of "None" to not filter out anything.
    frac_zero_reward_retained = None

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
    frac_zero_reward_retained: Optional[float],
    limit_samples: int,
    test_subset_within_epoch: Optional[int],
    opt_kwargs: Optional[Mapping[str, Any]],
    classify: bool,
    adversarial: bool,
    start_epoch: Optional[int],
    nonsense_reward: Optional[float],
    num_acts: Optional[int],
    vis_frac_per_epoch: Optional[float],
    gradient_clip_percentile: Optional[float],
    debugging: Mapping,
) -> SupervisedTrainer:
    if not adversarial:
        if not classify:
            # MSE loss with mean reduction (the default)
            # Mean reduction means every batch affects model updates the same,
            # regardless of batch_size.
            loss_fn = th.nn.MSELoss()
        else:
            # loss function takes outputs (interpreted as log-probability reward is
            # zero), reward, and computes the cross-entropy loss.
            def loss_fn(input, target):
                if len(input.shape) == 1:
                    input = input[:, None]
                zeros = th.zeros(input.shape)
                log_probs = th.cat((input, zeros), dim=1)
                target_classes = (target != 0).long()
                return th.nn.CrossEntropyLoss()(log_probs, target_classes)

    else:
        # Huber loss with mean reduction
        # When the prediction is within a distance of sqrt(3) of the regression target,
        # this is just equal to half of the MSE loss, otherwise it's L1 loss.
        # Designed to ensure that visualizations don't overwhelm the loss during
        # adversarial training
        loss_fn = th.nn.HuberLoss(delta=math.sqrt(3))

    trainer = SupervisedTrainer(
        demonstrations=expert_trajectories,
        limit_samples=limit_samples,
        reward_net=model,
        batch_size=batch_size,
        test_frac=test_frac,
        test_freq=test_freq,
        num_loader_workers=num_loader_workers,
        loss_fn=loss_fn,
        test_subset_within_epoch=test_subset_within_epoch,
        frac_zero_reward_retained=frac_zero_reward_retained,
        opt_kwargs=opt_kwargs,
        custom_logger=custom_logger,
        allow_variable_horizon=True,
        adversarial=adversarial,
        start_epoch=start_epoch,
        nonsense_reward=nonsense_reward,
        num_acts=num_acts,
        vis_frac_per_epoch=vis_frac_per_epoch,
        gradient_clip_percentile=gradient_clip_percentile,
        debug_settings=debugging,
    )
    return trainer

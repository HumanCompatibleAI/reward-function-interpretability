"""Common configuration elements for training supervised models."""

import logging

import sacred

supervised_ingredient = sacred.Ingredient("supervised")
logger = logging.getLogger(__name__)


@supervised_ingredient.config
def config():
    epochs = 100  # Number of training epochs
    test_frac = 0.1  # Fraction of training data to use for testing
    test_freq = 64  # Frequency of running tests (in batches)
    batch_size = 32  # Batch size for training a supervised model
    num_loader_workers = 0  # Number of workers for data loading

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
        # Log samples from the dataset in wandb. True to show all samples.
        show_samples=False,
        # If show_samples is True, log separate transitions as separate steps in wandb
        # if this is True, otherwise log them as separate entries in wandb.
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

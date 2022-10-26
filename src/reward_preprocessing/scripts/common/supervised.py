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

    locals()  # quieten flake8


@supervised_ingredient.config_hook
def config_hook(config, command_name, logger) -> dict:
    """Sets defaults for net_kwargs if not provided."""
    del command_name
    res = {}
    if (
        "use_done" in config["supervised"]["net_kwargs"]
        and config["supervised"]["net_kwargs"]["use_done"]
    ):
        logger.warning(
            "Supervised training does not support setting use_done to "
            "False. We don't support networks that take in the done signal. "
            "This value will be ignored."
        )

    return res

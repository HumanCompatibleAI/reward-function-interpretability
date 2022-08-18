"""Common configuration elements for training supervised models."""

import logging
from typing import Tuple

import sacred
from imitation.data.types import Transitions, transitions_collate_fn

from torch.utils import data as th_data

supervised_ingredient = sacred.Ingredient("supervised")
logger = logging.getLogger(__name__)


@supervised_ingredient.config
def config():
    epochs = 100  # Number of training epochs
    test_frac = 0.1  # Fraction of training data to use for testing
    test_freq = 64  # Frequency of running tests (in batches)
    batch_size = 32  # Batch size for training a supervised model
    num_loader_workers = 0  # Number of workers for data loading

    locals()  # quieten flake8

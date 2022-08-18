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


@supervised_ingredient.capture
def train_test_dataloaders(
    dataset: Transitions,
    test_frac: float,
    batch_size: int,
    num_loader_workers: int,
) -> Tuple[th_data.DataLoader, th_data.DataLoader]:
    """Split a dataset of transitions into train and test sets and returns dataloaders
    for each.

    Args:
        dataset: A dataset of transitions.
        test_frac: Fraction of the dataset to use for testing.
        batch_size: Batch size to return from dataloader.
        num_loader_workers: Number of workers to use for dataloader.
    """
    num_test = int(len(dataset) * test_frac)
    num_train = len(dataset) - num_test
    train, test = th_data.random_split(dataset, [num_train, num_test])
    train_loader = th_data.DataLoader(
        train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        collate_fn=transitions_collate_fn,
    )
    test_loader = th_data.DataLoader(
        test,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        collate_fn=transitions_collate_fn,
    )
    return train_loader, test_loader

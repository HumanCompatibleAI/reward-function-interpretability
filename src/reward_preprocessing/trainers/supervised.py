from typing import Optional, Sequence, Type, Mapping, Dict, Tuple

import torch as th
from imitation.algorithms import base
from imitation.data import types
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.data.types import transitions_collate_fn
from imitation.util import logger as imit_logger
from torch.utils import data as th_data

from reward_preprocessing.models import ProcgenCnnRegressionRewardNet


def _data_dict_to_model_args_and_target(
    data_dict: Dict[str, th.Tensor], device: str
) -> Tuple[tuple, th.Tensor]:
    """Move data to correct device and return for model args.

    Args:
        data_dict: Dictionary of data from Transitions dataloader to be passed to model.
        device: Device to move data to.
    """
    obs_bt = data_dict["obs"]
    act_bt = data_dict["acts"]
    next_obs_bt = data_dict["next_obs"]
    done_bt = data_dict["dones"]
    rew_bt = data_dict["rews"]

    obs = obs_bt.to(device)
    act = act_bt.to(device)
    next_obs = next_obs_bt.to(device)
    done = done_bt.to(device)
    target = rew_bt.to(device)

    return (obs, act, next_obs, done), target


class SupervisedTrainer(base.BaseImitationAlgorithm):
    """Learns from demonstrations (transitions / trajectories) using supervised
    learning. Has some overlap with base.DemonstrationAlgorithm, but does not train a
    policy."""

    def __init__(
        self,
        demonstrations: Sequence[types.TrajectoryWithRew],
        reward_net: ProcgenCnnRegressionRewardNet,
        batch_size: int,
        test_frac: float,
        test_freq: int,
        num_loader_workers: int,
        loss_fn,
        opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Mapping] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Creates an algorithm that learns from demonstrations.

        Args:
            demonstrations: Demonstrations from an expert (optional). Trajectories
                will be used as the dataset for supervised learning.
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
        """
        self._train_loader = None
        self._test_loader = None
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self._batch_size = batch_size
        self._test_frac = test_frac
        self._test_freq = test_freq
        self._num_loader_workers = num_loader_workers
        self._loss_fn = loss_fn

        self._reward_net = reward_net

        # Init optimizer
        self._opt = opt_cls(
            self._reward_net.parameters(),
            **opt_kwargs,
        )

        if demonstrations is not None:
            self.set_demonstrations(demonstrations)

    def set_demonstrations(
        self, demonstrations: Sequence[types.TrajectoryWithRew]
    ) -> None:
        """Sets train and test dataloaders from trajectories. Trajectories must contain
        reward data."""
        # Trajectories -> Transitions (both with reward)
        dataset = flatten_trajectories_with_rew(demonstrations)
        # Calculate the dataset split.
        num_test = int(len(dataset) * self._test_frac)
        num_train = len(dataset) - num_test
        train, test = th_data.random_split(dataset, [num_train, num_test])

        self._train_loader = th_data.DataLoader(
            train,
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            drop_last=True,
        )
        self._test_loader = th_data.DataLoader(
            test,
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            drop_last=True,
        )

    def train(self, num_epochs, device):
        for epoch in range(1, num_epochs + 1):
            self._train_batch(
                device,
                epoch,
            )

    def _train_batch(
        self,
        device,
        epoch,
    ):
        self._reward_net.train()
        for batch_idx, data_dict in enumerate(self._train_loader):
            model_args, target = _data_dict_to_model_args_and_target(data_dict, device)

            self._opt.zero_grad()
            output = self._reward_net(*model_args)
            loss = self._loss_fn(output, target)
            loss.backward()
            self._opt.step()
            if batch_idx % self._test_freq == 0:  # Test and log every test_freq batches
                test_loss = self._test(device, self._loss_fn)
                # description = (
                #     f"Epoch: {epoch}, train loss: {loss.item():.4f}, "
                #     f"test loss: {test_loss:.4f}"
                # )

    def _test(self, device, loss_fn) -> float:
        """Test model on data in test_loader. Returns average batch loss."""
        self._reward_net.eval()
        test_loss: th.Tensor = th.Tensor([0.0])
        with th.no_grad():
            for data_dict in self._test_loader:
                model_args, target = _data_dict_to_model_args_and_target(
                    data_dict, device
                )
                output = self._reward_net(*model_args)
                test_loss += loss_fn(output, target)  # sum up batch loss

        test_loss /= len(self._test_loader.dataset)
        self._reward_net.train()

        return test_loss.item()

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

from PIL import Image
from gym import spaces
from imitation.algorithms import base
from imitation.data import types
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.data.types import transitions_collate_fn
from imitation.rewards.reward_nets import RewardNet
from imitation.util import logger as imit_logger
import torch as th
from torch.utils import data
from tqdm import tqdm
import wandb


class SupervisedTrainer(base.BaseImitationAlgorithm):
    """Learns from demonstrations (transitions / trajectories) using supervised
    learning. Has some overlap with base.DemonstrationAlgorithm, but does not train a
    policy.
    """

    def __init__(
        self,
        demonstrations: Sequence[types.TrajectoryWithRew],
        reward_net: RewardNet,
        batch_size: int,
        test_frac: float,
        test_freq: int,
        num_loader_workers: int,
        loss_fn,
        opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Mapping] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        seed: Optional[int] = None,
    ):
        """Creates an algorithm that learns from demonstrations.

        Args:
            demonstrations: Demonstrations from an expert as trajectories with reward.
                Trajectories will be used as the dataset for supervised learning.
            reward_net:
                Reward network to train. This code assumes the net expects
                observations to be normalized between 0 and 1.
            batch_size: Batch size to use for training.
            test_frac: Fraction of dataset to use for testing.
            test_freq: Number of batches to train on before testing and logging.
            num_loader_workers: Number of workers to use for dataloader.
            loss_fn: Loss function to use for training.
            opt_cls: Optimizer class to use for training.
            opt_kwargs: Keyword arguments to pass to optimizer.
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
        self._global_batch_step = 0
        self._batch_size = batch_size
        self._test_frac = test_frac
        self._test_freq = test_freq
        self._num_loader_workers = num_loader_workers
        self._loss_fn = loss_fn

        self.reward_net = reward_net

        # Init optimizer
        self._opt = opt_cls(
            self.reward_net.parameters(),
            **opt_kwargs,
        )

        if demonstrations is not None:
            self.set_demonstrations(demonstrations, seed)

    def set_demonstrations(
        self,
        demonstrations: Sequence[types.TrajectoryWithRew],
        seed: Optional[int] = None,
    ) -> None:
        """Sets train and test dataloaders from trajectories. Trajectories must contain
        reward data."""
        # Trajectories -> Transitions (both with reward)
        dataset = flatten_trajectories_with_rew(demonstrations)
        # Calculate the dataset split.
        num_test = int(len(dataset) * self._test_frac)
        num_train = len(dataset) - num_test
        if seed is None:
            train, test = data.random_split(dataset, [num_train, num_test])
            shuffle_generator = None
        else:
            train, test = data.random_split(
                dataset,
                [num_train, num_test],
                generator=th.Generator().manual_seed(seed),
            )
            shuffle_generator = th.Generator().manual_seed(seed)

        self._train_loader = data.DataLoader(
            train,
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            drop_last=True,
            generator=shuffle_generator,
        )
        self._test_loader = data.DataLoader(
            test,
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            drop_last=True,
        )

    def train(
        self,
        num_epochs,
        device,
        callback: Optional[Callable[[int], None]] = None,
    ):
        """Trains the model on the data in train_loader.

        Args:
            num_epochs: Number of epochs to train for.
            device: Device to train on.
            callback: Optional callback to call after each epoch, takes the epoch number
                as the single argument (epoch numbers start at 1).
        """
        for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
            self._train_batch(
                device,
                epoch,
            )
            if callback is not None:
                callback(epoch)

    def _train_batch(
        self,
        device,
        epoch,
    ):
        """Trains the model on a single batch of data."""
        self.reward_net.train()
        sum_batch_losses = 0
        sample_count = 0
        for batch_idx, data_dict in enumerate(self._train_loader):
            self._global_batch_step += 1
            model_args, target = self._data_dict_to_model_args_and_target(
                data_dict, device
            )

            self._opt.zero_grad()
            output = self.reward_net(*model_args)
            loss = self._loss_fn(output, target)
            loss.backward()
            self._opt.step()
            sum_batch_losses += loss.item()
            sample_count += len(data_dict["obs"])
            if batch_idx % self._test_freq == 0:  # Test and log every test_freq batches
                self.logger.record("epoch", epoch)
                per_sample_loss = loss.item() / self._batch_size
                self.logger.record("train_loss", per_sample_loss)
                test_loss = self._test(device, self._loss_fn)
                self.logger.record("test_loss", test_loss)
                self.logger.dump(self._global_batch_step)

        # At the end of the epoch.
        per_sample_ep_loss = sum_batch_losses / sample_count
        self.logger.record("epoch_train_loss", per_sample_ep_loss)
        test_loss = self._test(device, self._loss_fn)
        self.logger.record("epoch_test_loss", test_loss)
        self._global_batch_step += 1  # dump() must be called with unique step.
        self.logger.dump(self._global_batch_step)

    def _test(self, device, loss_fn) -> float:
        """Test model on data in test_loader. Returns average batch loss."""
        self.reward_net.eval()
        test_loss = 0.0
        with th.no_grad():
            for data_dict in self._test_loader:
                model_args, target = self._data_dict_to_model_args_and_target(
                    data_dict, device
                )
                output = self.reward_net(*model_args)
                test_loss += loss_fn(output, target).item()  # sum up batch loss

        test_loss /= len(self._test_loader.dataset)  # Make it per-sample loss
        self.reward_net.train()

        return test_loss

    def _data_dict_to_model_args_and_target(
        self, data_dict: Dict[str, th.Tensor], device: str
    ) -> Tuple[tuple, th.Tensor]:
        """Turn data dict into structure that the model expects. Perform the following:
        - Normalize observations to be between 0 and 1.
            - Whether the data needs to be changed is determined by the type of
            the observations.
        - Turn actions into one-hot vectors.
        - Move data to correct device.
        - Return as tuple of args that can be passed to forward() and target.

        Args:
            data_dict: Dictionary of data from Transitions dataloader to be passed to
                model.
            device: Device to move data to.

        Returns:
            Tuple of model_args and target.
            modal_args is a tuple of the four model inputs: obs, next_obs, action, and
            done. Target is the target reward.
        """
        obs: th.Tensor = data_dict["obs"].to(device)
        act = data_dict["acts"].to(device)
        next_obs = data_dict["next_obs"].to(device)
        done = data_dict["dones"].to(device)
        target = data_dict["rews"].to(device)

        if obs.dtype == th.uint8:  # Observations saved as int => Normalize to [0, 1]
            obs = obs.float() / 255.0
        if next_obs.dtype == th.uint8:
            next_obs = next_obs.float() / 255.0

        if isinstance(self.reward_net.action_space, spaces.Discrete):
            num_actions = self.reward_net.action_space.n
        else:
            raise NotImplementedError("Trainer only supports discrete action spaces.")
        if act.dtype == th.float:
            self.logger.warn("Actions are of float type. Converting to int.")
            # long necessary in order to use integer action as index for one-hot vector.
            act = act.long()
        act = th.nn.functional.one_hot(act, num_actions)

        return (obs, act, next_obs, done), target

    def log_data_stats(self):
        """Logs data statistics to logger."""

        self.logger.log("Calculating stats for train data...")
        self._record_dataset_stats("train_data", self._train_loader)
        self.logger.log("Calculating stats for test data...")
        self._record_dataset_stats("test_data", self._test_loader)

    def _record_dataset_stats(self, key: str, dataloader: data.DataLoader) -> None:
        """Calculate useful statistics about a dataset.
        Calculates
        - size of the dataset.
        - mean and standard deviation of observations.
        - mean and standard deviation of rewards.
        - histogram of the rewards.
        - histogram of actions.
        """
        sample_count = 0
        # Holds the mean of each channel for every sample.
        obs_reduced = []
        rewards = []
        actions = []
        dones_count = 0
        for batch_idx, data_dict in enumerate(dataloader):
            obs = data_dict["obs"]
            rew = data_dict["rews"]
            act = data_dict["acts"]
            done = data_dict["dones"]
            sample_count += obs.shape[0]
            # Dim 0 is for batch, dim 3 is for channels.
            obs_reduced.append(th.mean(obs, dim=[1, 2]))
            rewards.append(rew)
            actions.append(act)
            dones_count += th.sum(done).item()
        obs_tensor = th.cat(obs_reduced, dim=0)
        rew_tensor = th.cat(rewards, dim=0)
        act_tensor = th.cat(actions, dim=0)
        obs_std, obs_mean = th.std_mean(obs_tensor, dim=0)
        rew_std, rew_mean = th.std_mean(rew_tensor, dim=0)

        rew_hist = wandb.Histogram(rew_tensor, num_bins=11)
        act_hist = wandb.Histogram(act_tensor, num_bins=16)

        # Record the calculated statistics.
        self.logger.record(f"{key}/size", sample_count)
        for channel_i in range(len(obs_mean)):
            self.logger.record(
                f"{key}/obs_mean_channel{channel_i}", obs_mean[channel_i]
            )
            self.logger.record(f"{key}/obs_std_channel{channel_i}", obs_std[channel_i])
        self.logger.record(f"{key}/rew_mean", rew_mean)
        self.logger.record(f"{key}/rew_std", rew_std)
        self.logger.record(f"{key}/rew_hist", rew_hist)
        self.logger.record(f"{key}/act_hist", act_hist)
        self.logger.record(f"{key}/done_mean", dones_count / sample_count)

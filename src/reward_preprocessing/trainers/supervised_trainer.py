import math
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

from gym import spaces
from imitation.algorithms import base
from imitation.data import types
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.data.types import transitions_collate_fn
from imitation.rewards.reward_nets import RewardNet
from imitation.util import logger as imit_logger
import numpy as np
import torch as th
from torch.utils import data
from tqdm import tqdm
import wandb

from reward_preprocessing.common.utils import (
    TensorTransitionWrapper,
    TransformedDataset,
    log_img_wandb,
    make_transition_to_tensor,
    tensor_to_transition,
)
from reward_preprocessing.vis.reward_vis import LayerNMF


def _normalize_obs(obs: th.Tensor) -> th.Tensor:
    """Normalize by dividing by 255, if obs is uint8, otherwise no change."""
    if obs.dtype == th.uint8:  # Observations saved as int => Normalize to [0, 1]
        obs = obs.float() / 255.0
    return obs.float()  # This ensures we have float tensors and not double tensors.


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
        loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
        frac_zero_reward_retained: Optional[float] = None,
        limit_samples: int = -1,
        test_subset_within_epoch: Optional[int] = None,
        opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Mapping[str, Any]] = None,
        adversarial: bool = False,
        nonsense_reward: Optional[float] = None,
        num_acts: Optional[int] = None,
        vis_frac_per_epoch: Optional[float] = None,
        gradient_clip_percentile: Optional[float] = None,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        seed: Optional[int] = None,
        debug_settings: Optional[Mapping[str, Any]] = None,
    ):
        """Creates an algorithm that learns from demonstrations.

        Args:
            demonstrations:
                Demonstrations from an expert as trajectories with reward.
                Trajectories will be used as the dataset for supervised learning.
            reward_net:
                Reward network to train. This code assumes the net expects
                observations to be normalized between 0 and 1.
            batch_size: Batch size to use for training.
            test_frac: Fraction of dataset to use for testing.
            test_freq: Number of batches to train on before testing and logging.
            num_loader_workers: Number of workers to use for dataloader.
            loss_fn:
                Loss function to use for training. Function should not be averaged over
                the batch, but accumulated over the batch. This is because batches
                might have different sizes. SupervisedTrainer will normalize the
                loss per sample (i.e. per transition) for logging.
            frac_zero_reward_retained:
                If not None, remove 1 - (this fraction) of training examples where the
                reward is zero, to manually re-weight the dataset to non-zero reward
                transitions.
            limit_samples: If positive, only use this many samples from the dataset.
            test_subset_within_epoch:
                If not none, only use this many test batches when evaluating test loss
                in the middle of a training epoch.
            opt_cls: Optimizer class to use for training.
            opt_kwargs: Keyword arguments to pass to optimizer.
            adversarial: Train on adversarial examples (aka network visualizations).
            nonsense_reward: Reward to assign to adversarial examples.
            num_acts: Number of acts the network can take.
            vis_frac_per_epoch:
                How many adversarial examples to add to the train set per epoch,
                expressed as a fraction of the original train set.
            gradient_clip_percentile:
                If doing adversarial training, the percentile of norms of first epoch
                gradients that we clip gradients of later epochs (that include
                high-loss adversarial examples) to.
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon:
                If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
            debug_settings: Dictionary of various debug settings.
        """
        self._train_loader = None
        self._test_loader = None
        self._train_set = None
        self._shuffle = None
        self._shuffle_generator = None
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self._global_batch_step = 0
        self._batch_size = batch_size
        self._test_frac = test_frac
        self._test_freq = test_freq
        self._test_subset_within_epoch = test_subset_within_epoch
        self._num_loader_workers = num_loader_workers
        self._loss_fn = loss_fn

        if frac_zero_reward_retained is not None:
            if frac_zero_reward_retained < 0.0 or frac_zero_reward_retained > 1.0:
                raise ValueError(
                    "frac_zero_reward_retained should be between 0 and 1, is set to"
                    + f"{frac_zero_reward_retained}"
                )

        self._frac_zero_reward_retained = frac_zero_reward_retained

        self.reward_net = reward_net

        self.debug_settings = {} if debug_settings is None else debug_settings

        # Init optimizer
        self._opt = opt_cls(
            self.reward_net.parameters(),
            **opt_kwargs,
        )

        self.limit_samples = limit_samples

        self.adversarial = adversarial
        if self.adversarial:
            if nonsense_reward is None:
                raise ValueError(
                    "Must provide reward value for adversarially generated"
                    + " inputs as the 'nonsense_value' argument."
                )
            if vis_frac_per_epoch is None:
                raise ValueError(
                    "Must specify how many visualizations to add per epoch as a "
                    + "fraction of the train set size, as the 'vis_frac_per_epoch' "
                    + "argument."
                )
            if vis_frac_per_epoch < 0.0 or vis_frac_per_epoch > 1.0:
                raise ValueError(
                    "vis_frac_per_epoch should be between 0 and 1, but is set as "
                    + f"{vis_frac_per_epoch}"
                )
            if num_acts is None:
                raise ValueError(
                    "Must specify how many actions are available in this"
                    + " environment as the 'num_acts' argument."
                )
            if gradient_clip_percentile is None:
                raise ValueError(
                    "Must specify what percentile of first-epoch gradient norms to clip"
                    + " later epoch gradient norms to."
                )
            if gradient_clip_percentile < 0.0 or gradient_clip_percentile > 1.0:
                raise ValueError(
                    "gradient_clip_percentile should be between 0 and 1, but is set as "
                    + f"{gradient_clip_percentile}"
                )
            self.nonsense_reward = nonsense_reward
            self.vis_frac_per_epoch = vis_frac_per_epoch
            self.wrapped_reward_net = TensorTransitionWrapper(self.reward_net)
            self.num_acts = num_acts
            self.gradient_clip_percentile = gradient_clip_percentile

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
        if self._frac_zero_reward_retained is not None:
            # randomly filter out zero-reward transitions (manual reweighting)
            started = False
            for datum in dataset:
                if (
                    datum["rews"] != 0.0
                    or random.random() < self._frac_zero_reward_retained
                ):
                    if not started:
                        new_obs = datum["obs"][None, :, :, :]
                        new_acts = [datum["acts"]]
                        new_next_obs = datum["next_obs"][None, :, :, :]
                        new_infos = [datum["infos"]]
                        new_dones = [datum["dones"]]
                        new_rews = [datum["rews"]]
                        started = True
                    else:
                        new_obs = np.concatenate(
                            [new_obs, datum["obs"][None, :, :, :]], axis=0
                        )
                        new_acts.append(datum["acts"])
                        new_next_obs = np.concatenate(
                            [new_next_obs, datum["next_obs"][None, :, :, :]], axis=0
                        )
                        new_infos.append(datum["infos"])
                        new_dones.append(datum["dones"])
                        new_rews.append(datum["rews"])

            dataset = types.TransitionsWithRew(
                obs=np.array(new_obs),
                acts=np.array(new_acts),
                infos=np.array(new_infos),
                next_obs=np.array(new_next_obs),
                dones=np.array(new_dones),
                rews=np.array(new_rews),
            )
            print("done filtering dataset")
        if self.limit_samples == 0:
            raise ValueError("Can't train on 0 samples")
        elif self.limit_samples > 0:
            # Instead of taking the first `limit_samples` samples, we take the last,
            # to increase the change that samples include rewards from the end of the
            # trajectory.
            dataset = dataset[-self.limit_samples :]
        # Calculate the dataset split.
        num_test = int(len(dataset) * self._test_frac)
        if num_test <= 0:
            raise ValueError("Test fraction too small, would result in empty test set")
        num_train = len(dataset) - num_test

        # Usually we always shuffle. This concerns both the dataset split and the
        # shuffling when loading data from the Dataloader.
        # If this debug setting is set we disable both shuffling during split and during
        # data loading. Test loader will always be deterministic.
        shuffle = not self.debug_settings.get("disable_dataset_shuffling", False)

        if shuffle:
            if seed is None:
                train, test = data.random_split(dataset, [num_train, num_test])
                shuffle_generator = None
            else:
                shuffle_generator = th.Generator().manual_seed(seed)
                train, test = data.random_split(
                    dataset,
                    [num_train, num_test],
                    generator=shuffle_generator,
                )
        else:
            # Debug setting with disabled shuffling: Non-random split.
            train = data.Subset(dataset, range(num_train))
            test = data.Subset(dataset, range(num_train, num_train + num_test))
            # Not needed, since shuffling is disabled anyway.
            shuffle_generator = None

        assert len(train) > 0, "Train set is empty"
        assert len(test) > 0, "Test set is empty"
        assert (
            len(train) == num_train
        ), f"Train set has wrong length. Is {len(train)}, should be {num_train}"
        assert (
            len(test) == num_test
        ), f"Test set has wrong length. Is {len(test)}, should be {num_test}"

        self._train_set = train
        self._shuffle = shuffle
        self._shuffle_generator = shuffle_generator

        self._train_loader = data.DataLoader(
            self._train_set,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            generator=self._shuffle_generator,
        )
        self._test_loader = data.DataLoader(
            test,
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
        )

        if self.adversarial:
            # Figure out how many adversarial examples to add per epoch.
            self.num_vis_per_epoch = int(num_train * self.vis_frac_per_epoch)
            if self.num_vis_per_epoch < 1:
                raise ValueError(
                    "vis_frac_per_epoch is set too low: its value is "
                    + f"{self.vis_frac_per_epoch}, and at that value "
                    + f"{self.num_vis_per_epoch} adversarial examples will be added "
                    + "per epoch. vis_frac_per_epoch should be at least 1 / (size of "
                    + f"train set), and train set has size {num_train}."
                )
            # Generate data for pre-processing for adversarial example purposes.
            tensor_transitions = TransformedDataset(
                dataset, make_transition_to_tensor(self.num_acts)
            )
            transitions_dataloader = data.DataLoader(
                tensor_transitions, shuffle=True, batch_size=self._batch_size
            )
            trans_tens_batch = next(iter(transitions_dataloader))
            self.trans_tens_batch = trans_tens_batch.float()

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
                Will usually save the network at certain epochs. If network is being
                adversarially trained, should also save the latest network
                visualizations.
        """
        self._logger.log(f"Using optimizer {self._opt}")
        self._logger.log(f"Using loss function {self._loss_fn}")
        self._logger.log("Starting training")

        # Determine test and train loss once before training starts.
        train_loss = self._eval_on_dataset(device, self._loss_fn, self._train_loader)
        self.logger.record("epoch_train_loss", train_loss)
        test_loss = self._eval_on_dataset(device, self._loss_fn, self._test_loader)
        self.logger.record("epoch_test_loss", test_loss)
        # Both will be logged as the 0th batch.
        self.logger.dump(self._global_batch_step)

        for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
            self._train_batch(
                device,
                epoch,
            )
            if self.adversarial:
                self._add_adversarial_inputs(epoch, device)
            if callback is not None:
                callback(epoch)

    def _add_adversarial_inputs(self, epoch: int, device):
        """Generates inputs that max reward_net output, adds them to train data."""
        # Generate dataset of adversarial examples / visualizations.
        vis_obs = []
        vis_acts = []
        vis_next_obs = []
        num_vis_calls = (
            self.num_vis_per_epoch
            if not self.reward_net.use_action
            else math.ceil(self.num_vis_per_epoch / self.num_acts)
        )
        for i in range(num_vis_calls):
            obs, acts, next_obs = self._visualize_network(device)
            if i == 0:
                vis_obs = obs.detach().cpu().numpy()
                vis_acts = acts.detach().cpu().numpy()
                vis_next_obs = next_obs.detach().cpu().numpy()
            else:
                vis_obs = np.concatenate([vis_obs, obs.detach().cpu().numpy()], axis=0)
                vis_acts = np.concatenate(
                    [vis_acts, acts.detach().cpu().numpy()], axis=0
                )
                vis_next_obs = np.concatenate(
                    [vis_next_obs, next_obs.detach().cpu().numpy()], axis=0
                )
        # Turn them into TransitionsWithRew.
        dones = np.array([False] * vis_obs.shape[0])
        infos = np.array([{}] * vis_obs.shape[0])
        rews = np.array([self.nonsense_reward] * vis_obs.shape[0]).astype(np.float32)
        vis_dataset = types.TransitionsWithRew(
            obs=vis_obs,
            acts=vis_acts,
            next_obs=vis_next_obs,
            dones=dones,
            infos=infos,
            rews=rews,
        )

        # below is used to save the visualizations (in scripts/train_regression.py)
        self.latest_visualizations = vis_dataset
        # Add to the train dataset.
        self._train_set = data.ConcatDataset([self._train_set, vis_dataset])
        self._train_loader = data.DataLoader(
            self._train_set,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
            num_workers=self._num_loader_workers,
            collate_fn=transitions_collate_fn,
            generator=self._shuffle_generator,
        )

    def _visualize_network(self, device):
        num_features = self.num_acts if self.reward_net.use_action else 1

        nmf = LayerNMF(
            model=self.wrapped_reward_net,
            features=num_features,
            layer_name="rew_net_cnn_dense_final",
            model_inputs_preprocess=self.trans_tens_batch.to(device),
            activation_fn="relu",
        )
        visualization_np = nmf.vis_traditional()
        visualization_tens = th.tensor(visualization_np)
        visualization_tens = th.permute(visualization_tens, (0, 3, 1, 2))
        obs, acts, next_obs = tensor_to_transition(visualization_tens)
        # There isn't ever any gradient to the actions, which are just used (if at all)
        # to select which final-layer neuron to read out reward from.
        # So, if actions are meaningful at all, they should go sequentially,
        # since in that eventuality each transition will be chosen to maximize the
        # reward conditioned on some variable action, that action varying sequentially.
        action_nums = th.tensor(list(range(num_features))).to(device)
        return obs, action_nums, next_obs

    def _train_batch(
        self,
        device,
        epoch,
    ):
        """Trains the model on a single batch of data."""
        self.reward_net.train()
        weighted_batch_losses = 0
        sample_count = 0
        # For adversarial training, find the (self.gradient_clip_percentile)th
        # percentile gradient norm in the first epoch, and clip future gradients to
        # that.
        # This avoids gradients exploding due to extremely large mis-predictions on
        # adversarial examples.
        if self.adversarial and epoch == 1:
            grad_norms = []

        for batch_idx, data_dict in enumerate(self._train_loader):
            self._global_batch_step += 1
            model_args, target = self._data_dict_to_model_args_and_target(
                data_dict, device
            )
            self._opt.zero_grad()
            output = self.reward_net(*model_args)
            loss = self._loss_fn(output, target)
            loss.backward()

            if self.adversarial and epoch == 1:
                grads = [
                    param.grad.detach().flatten()
                    for param in self.reward_net.parameters()
                    if param.grad is not None
                ]
                grad_norm = th.cat(grads).norm()
                grad_norms.append(grad_norm)
            if self.adversarial and epoch > 1:
                # clip norm
                th.nn.utils.clip_grad_norm_(
                    self.reward_net.parameters(), self._grad_clip_val
                )

            self._opt.step()
            # Weigh each loss by the number of samples in the batch. This way we can
            # divide by the number of samples to get the average per-sample loss.
            weighted_batch_losses += loss.item() * len(target)
            sample_count += len(target)
            if batch_idx % self._test_freq == 0:  # Test and log every test_freq batches
                self.logger.record("epoch", epoch)
                # Log the mean loss over the batch.
                self.logger.record("train_loss", loss.item())
                # Determine the mean loss over (a subset of) the test dataset.
                test_loss = self._eval_on_dataset(
                    device,
                    self._loss_fn,
                    self._test_loader,
                    num_iters=self._test_subset_within_epoch,
                )
                self.logger.record("test_loss", test_loss)
                self.logger.dump(self._global_batch_step)

        # At the end of the epoch.
        if self.adversarial and epoch == 1:
            grad_norms.sort()
            percentile = int(len(grad_norms) * self.gradient_clip_percentile)
            self._grad_clip_val = grad_norms[percentile]
        per_sample_ep_loss = weighted_batch_losses / sample_count
        self.logger.record("epoch_train_loss", per_sample_ep_loss)
        test_loss = self._eval_on_dataset(device, self._loss_fn, self._test_loader)
        self.logger.record("epoch_test_loss", test_loss)
        self._global_batch_step += 1  # dump() must be called with unique step.
        self.logger.dump(self._global_batch_step)

    def _eval_on_dataset(
        self,
        device: str,
        loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
        dataloader: th.utils.data.DataLoader,
        num_iters: Optional[int] = None,
    ) -> float:
        """Evaluate model on provided data loader. Returns loss, averaged over the
        number of samples in the dataset. Model is set to eval mode before evaluation
        and back to train mode afterwards. Set num_iters to only evaluate on a subset.
        """
        self.reward_net.eval()
        weighted_test_loss = 0.0
        # Determine number of items in the dataloader manually, since not every
        # dataloader has a .dataset which supports len() (AFAICT).
        # Also: If dataloader truncates, there are fewer items being used for evaluation
        # than there are in the full (un-truncated) dataset.
        num_items = 0
        i = 0
        with th.no_grad():
            for data_dict in dataloader:
                model_args, target = self._data_dict_to_model_args_and_target(
                    data_dict, device
                )
                output = self.reward_net(*model_args)
                # Sum up batch loss
                weighted_test_loss += loss_fn(output, target).item() * len(target)
                num_items += len(target)  # Count total number of samples
                i += 1
                if num_iters is not None and i == num_iters:
                    # break out of loop early if we don't want to loop over whole test
                    # set
                    break

        sample_test_loss = weighted_test_loss / num_items  # Make it per-sample loss
        self.reward_net.train()

        return sample_test_loss

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

        obs = _normalize_obs(obs)
        next_obs = _normalize_obs(next_obs)

        if isinstance(self.reward_net.action_space, spaces.Discrete):
            num_actions = self.reward_net.action_space.n
        else:
            raise NotImplementedError("Trainer only supports discrete action spaces.")

        if act.dtype != th.long:
            if act.dtype == th.float:
                # Discrete actions should really be saved as integer values, so we
                # give a warning.
                self.logger.warn("Actions are of float type. Converting to int.")
            # Convert to long. This is necessary in order to use integer action
            # as index for one-hot vector.
            # However, unlike above no warning is logged because saving actions as int
            # is fine.
            act = act.long()
        act = th.nn.functional.one_hot(act, num_actions)

        return (obs, act, next_obs, done), target

    def log_data_stats(self):
        """Logs data statistics to logger."""

        self.logger.log("Calculating stats for train data...")
        self._record_stats_for_data("train_data", self._train_loader)
        self.logger.log("Calculating stats for test data...")
        self._record_stats_for_data("test_data", self._test_loader)

    def _record_stats_for_data(self, name: str, dataloader: data.DataLoader) -> None:
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
            obs = _normalize_obs(obs)

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

        # Collect non-zero rewards.
        non_zero_rew = rew_tensor[rew_tensor != 0]
        num_non_zero_rew = non_zero_rew.shape[0]

        obs_std, obs_mean = th.std_mean(obs_tensor, dim=0)
        rew_std, rew_mean = th.std_mean(rew_tensor, dim=0)
        non_zero_rew_std, non_zero_rew_mean = th.std_mean(non_zero_rew, dim=0)
        non_zero_rew_median = th.median(non_zero_rew)

        rew_hist = wandb.Histogram(rew_tensor, num_bins=11)
        act_hist = wandb.Histogram(act_tensor.float(), num_bins=16)

        # Record the calculated statistics.
        # New version of sb3 doesn't allow duplicate keys after the slash so we also
        # need to add the name there.
        self.logger.record(f"{name}/size_{name}", sample_count)
        for channel_i in range(len(obs_mean)):
            self.logger.record(
                f"{name}/obs_mean_channel{channel_i}_{name}", obs_mean[channel_i]
            )
            self.logger.record(
                f"{name}/obs_std_channel{channel_i}_{name}", obs_std[channel_i]
            )
        self.logger.record(f"{name}/rew_mean_{name}", rew_mean)
        self.logger.record(f"{name}/rew_std_{name}", rew_std)
        self.logger.record(f"{name}/rew_hist_{name}", rew_hist)
        self.logger.record(f"{name}/act_hist_{name}", act_hist)
        self.logger.record(f"{name}/done_mean_{name}", dones_count / sample_count)
        self.logger.record(f"{name}/done_count_{name}", dones_count)
        self.logger.record(f"{name}/non_zero_rew_mean_{name}", non_zero_rew_mean)
        self.logger.record(f"{name}/non_zero_rew_std_{name}", non_zero_rew_std)
        self.logger.record(f"{name}/non_zero_rew_median_{name}", non_zero_rew_median)
        self.logger.record(f"{name}/non_zero_rew_count_{name}", num_non_zero_rew)

    def log_samples(self, log_as_step: bool = False):
        obs_list = []  # To collect all observations to turn into video.
        count = 0
        for data_dict in self._train_loader:
            (
                (
                    obs,
                    act,
                    next_obs,
                    done,
                ),
                target,
            ) = self._data_dict_to_model_args_and_target(data_dict, "cpu")
            obs: th.Tensor
            next_obs: th.Tensor
            obs_list.append(obs)
            for i in range(len(obs)):
                reward = target[i].item()
                # Concatenate obs and next_obs to make a single image of the transition.
                img = np.concatenate([obs[i].numpy(), next_obs[i].numpy()], axis=1)
                if log_as_step:
                    wandb_key = "transition"
                    step = count
                else:
                    wandb_key = f"transition_{count}"
                    step = None

                log_img_wandb(
                    img=img,
                    logger=self.logger,
                    caption=f"Reward {reward}",
                    wandb_key=wandb_key,
                    scale=4,
                    step=step,  # Dump all images together with the first step.
                )
                count += 1

        try:
            # We only import these here to check whether these packages are installed.
            # wandb.Video requires them.
            # However, we don't want them as a hard requirement.
            # pytype: disable=import-error
            import imageio  # noqa: F401
            import moviepy  # noqa: F401

            # pytype: enable=import-error
            # Turn transitions into video.
            obs_tensor = th.cat(obs_list)
            # Vid expects channels first.
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            frames = np.uint8(obs_tensor.numpy() * 255)
            self.logger.record("traj_vid", wandb.Video(frames, fps=12))
        except ImportError:
            self.logger.warn(
                "moviepy or imageio not installed. Not logging transitions as video "
                "animation for debugging purposes. If you want to, run "
                "'pip install moviepy imageio'."
            )

        if log_as_step:
            step = count
        else:
            step = 0
        self.logger.dump(step=step)

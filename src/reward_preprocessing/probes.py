"""Train linear probes on reward nets."""

import math
import random
from typing import List, Optional, Tuple, Union

from imitation.rewards.reward_nets import CnnRewardNet, cnn_transpose
from stable_baselines3.common import preprocessing
import torch as th
from torch import nn

from reward_preprocessing.common.utils import (
    DoubleInfoTransitionsWithRew,
    transitions_collate_fn,
)


class Probe(nn.Module):
    # inspired by
    # https://github.com/yukimasano/linear-probes/blob/master/eval_linear_probes.py
    # TODO: remove magic constants for numbers of channels
    # TODO: change name to CnnProbe???
    # TODO: add check on attribute_dim
    # TODO: would be nice to reuse reward_net methods to a greater extent than I
    # currently am able to.
    def __init__(
        self,
        reward_net: CnnRewardNet,
        layer_name: str,
        attribute_dim: int,
        attribute_name: Union[str, List[str]],
        loss_type: str,
        device: th.device,
    ) -> None:
        super(Probe, self).__init__()
        self.attribute_name = attribute_name
        self.attribute_dim = attribute_dim
        self.model = reward_net.cnn
        self.use_state = reward_net.use_state
        self.use_action = reward_net.use_action
        self.use_next_state = reward_net.use_next_state
        self.use_done = reward_net.use_done
        self.observation_space = reward_net.observation_space
        self.normalize_inputs = reward_net.normalize_images
        self.hwc_format = reward_net.hwc_format
        self.layer_name = layer_name
        self.probe_head = None
        self.loss_type = loss_type
        self.device = device

        if self.loss_type not in ["mse", "cross_entropy"]:
            raise ValueError(
                f"loss_type was {self.loss_type}, should be mse or cross_entropy"
            )
        elif self.loss_type == "mse":
            self.loss_func = nn.MSELoss()
        else:
            assert self.loss_type == "cross_entropy"
            self.loss_func = nn.CrossEntropyLoss()

        input_channels = 0

        if self.use_state:
            input_channels += 3

        if self.use_next_state:
            input_channels += 3

        if not (self.use_state or self.use_next_state):
            raise ValueError("Reward net must use state or next_state")

        self.model.to(self.device)

        x = th.zeros(1, input_channels, 64, 64).to(device)
        # man I wish I had broken out the function that took sas' to a tensor
        # when I was writing CnnRewardNet.
        for name, child in self.model.named_children():
            x = child.forward(x)
            if name == self.layer_name:
                avg_pool = nn.AdaptiveAvgPool2d(1)
                flatten = nn.Flatten()
                fc = nn.Linear(x.size(1), attribute_dim)
                self.probe_head = nn.Sequential(avg_pool, flatten, fc)
        if self.probe_head is None:
            raise ValueError(f"Could not find layer {self.layer_name} to probe")

        self.probe_head.to(self.device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.model.eval()
        for name, child in self.model.named_children():
            x = child.forward(x)
            if name == self.layer_name:
                return self.probe_head(x)
        assert False, f"Could not find layer {self.layer_name} to probe."

    def train(
        self,
        dataset: DoubleInfoTransitionsWithRew,
        lr: float,
        frac_train: float,
        batch_size: int,
        num_epochs: int,
    ) -> None:
        """Train a probe on a dataset using stochastic gradient descent.

        Args:
          - dataset: collection of transitions (with info dicts for the obs and the
            next_obs).
          - lr: learning rate.
          - frac_train: proportion of the dataset to train on. remainder is a held-out
            test set.
          - batch_size: batch size.
          - num_epochs: number of epochs to train for.
        """
        train_loader, test_loader = self.make_loaders(dataset, frac_train, batch_size)
        init_train_loss = self.eval_on_dataloader(train_loader)
        print("Initial train loss:", init_train_loss)
        init_test_loss = self.eval_on_dataloader(test_loader)
        print("Initial test loss:", init_test_loss)
        optimizer = th.optim.SGD(self.probe_head.parameters(), lr=lr)
        for epoch in range(num_epochs):
            epoch_loss = self.eval_on_dataloader(train_loader, optimizer=optimizer)
            print(f"Train loss over epoch {epoch}:", epoch_loss)
            epoch_test_loss = self.eval_on_dataloader(test_loader)
            print(f"Test loss after epoch {epoch}:", epoch_test_loss)
        print("Training complete!")

    def make_loaders(
        self,
        dataset: DoubleInfoTransitionsWithRew,
        frac_train: float,
        batch_size: int,
    ) -> Tuple[th.utils.data.DataLoader, th.utils.data.DataLoader]:
        shuffled_dataset = random.sample(dataset, len(dataset))
        num_train = math.floor(frac_train * len(dataset))
        train_data = shuffled_dataset[:num_train]
        test_data = shuffled_dataset[num_train:]
        train_loader = th.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=transitions_collate_fn,
        )
        test_loader = th.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=transitions_collate_fn,
        )
        return train_loader, test_loader

    def data_dict_to_args_and_target(
        self, data_dict: dict
    ) -> Tuple[th.Tensor, th.Tensor]:
        info_dicts = (
            data_dict["next_infos"] if self.use_next_state else data_dict["infos"]
        )
        target = (
            th.tensor(
                [
                    [info_dict[name] for name in self.attribute_name]
                    for info_dict in info_dicts
                ]
            ).to(th.float32)
            if isinstance(self.attribute_name, list)
            else th.tensor(
                [info_dict[self.attribute_name] for info_dict in info_dicts]
            ).to(th.float32)
        )
        obses = data_dict["obs"]
        next_obses = data_dict["next_obs"]
        if self.hwc_format:
            obses = cnn_transpose(obses)
            next_obses = cnn_transpose(next_obses)
        obses = preprocessing.preprocess_obs(
            obses, self.observation_space, self.normalize_inputs
        )
        next_obses = preprocessing.preprocess_obs(
            next_obses, self.observation_space, self.normalize_inputs
        )
        args = None
        if self.use_state and self.use_next_state:
            args = th.concat([obses, next_obses], axis=1)
        elif self.use_state and not self.use_next_state:
            args = obses
        elif (not self.use_state) and self.use_next_state:
            args = next_obses
        else:
            assert False, "either use_state or use_next_state should have been True"
        return args, target

    def eval_on_dataloader(
        self,
        data_loader: th.utils.data.DataLoader,
        optimizer: Optional[th.optim.Optimizer] = None,
    ) -> float:
        """Evaluates the probe on a data loader, and optionally trains it.

        If optimizer is not None, it trains the model (and evaluates it in train mode),
        otherwise the model is run in eval mode.
        """
        self.model.eval()
        if optimizer is not None:
            self.probe_head.train()
        else:
            self.probe_head.eval()
        total_loss = 0.0
        num_batches = 0
        for data in data_loader:
            if optimizer is not None:
                optimizer.zero_grad()
            args, target = self.data_dict_to_args_and_target(data)
            args = args.to(self.device)
            target = target.to(self.device)
            outputs = self.forward(args)
            loss = self.loss_func(outputs, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

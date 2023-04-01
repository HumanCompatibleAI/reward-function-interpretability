"""Train linear probes on reward nets."""

import math
from typing import Optional, Tuple, Union

from imitation.rewards.reward_nets import CnnRewardNet
import torch as th
from torch import nn

from reward_preprocessing.common.utils import DoubleInfoTransitionsWithRew


class Probe(nn.Module):
    # inspired by
    # https://github.com/yukimasano/linear-probes/blob/master/eval_linear_probes.py
    # when you optimize, can just optimize over probe_head.params
    # TODO: remove magic constants for numbers of channels
    # TODO: change name to CnnProbe???
    # TODO: add check on attribute_dim
    def __init__(
        self,
        reward_net: CnnRewardNet,
        layer_name: str,
        attribute_dim: int,
        attribute_name: Union[str, list[str]],
        loss_type: str,
    ) -> None:
        super(Probe, self).__init__()
        self.attribute_name = attribute_name
        self.attribute_dim = attribute_dim
        self.model = reward_net.cnn
        self.use_state = reward_net.use_state
        self.use_action = reward_net.use_action
        self.use_next_state = reward_net.use_next_state
        self.use_done = reward_net.use_done
        self.layer_name = layer_name
        self.probe_head = None
        self.loss_type = loss_type

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

        x = th.zeros(1, input_channels, 64, 64)
        # man I wish I had broken out the function that took sas' to a tensor
        # when I was writing CnnRewardNet.
        for name, child in enumerate(self.model.named_children()):
            x = child.forward(x)
            if name == self.layer_name:
                avg_pool = nn.AdaptiveAvgPool2d(1)
                flatten = nn.Flatten()
                fc = nn.Linear(x.size(1), attribute_dim)
                self.probe_head = nn.Sequential(avg_pool, flatten, fc)
        if self.probe_head is None:
            raise ValueError(f"Could not find layer {self.layer_name} to probe")

    def forward(self, x: th.Tensor) -> th.Tensor:
        self.model.eval()
        for name, child in enumerate(self.model.named_children()):
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
        device = "cuda" if th.cuda.is_available() else "cpu"
        self.model.to(device)
        self.probe_head.to(device)
        train_loader, test_loader = self.make_loaders(dataset, frac_train, batch_size)
        init_train_loss = self.eval_on_dataloader(train_loader)
        print("Initial train loss:", init_train_loss)
        init_test_loss = self.eval_on_dataloader(test_loader)
        print("Initial test loss:", init_test_loss)
        optimizer = th.optim.SGD(self.probe_head.parameters(), lr=lr)
        for epoch in range(num_epochs):
            epoch_loss = self.eval_on_dataloader(train_loader, optimizer=optimizer)
            print(f"Average loss over epoch {epoch}:", epoch_loss)
            epoch_test_loss = self.eval_on_dataloader(test_loader)
            print(f"Test loss after epoch {epoch}:", epoch_test_loss)
        print("Training complete!")

    def make_loaders(
        self,
        dataset: DoubleInfoTransitionsWithRew,
        frac_train: float,
        batch_size: int,
    ) -> Tuple[th.utils.data.DataLoader, th.utils.data.DataLoader]:
        # TODO shuffle the dataset
        shuffled_dataset = dataset
        num_train = math.floor(frac_train * len(dataset))
        train_data = shuffled_dataset[:num_train]
        test_data = shuffled_dataset[num_train:]
        # wait probably these lists are enough to be th.utils.data.Datasets
        train_loader = th.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_loader = th.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )
        return train_loader, test_loader

    def data_dict_to_args_and_target(
        self, data_dict: dict
    ) -> Tuple[th.Tensor, th.Tensor]:
        info_dict = (
            data_dict["next_infos"] if self.use_next_state else data_dict["infos"]
        )
        target = (
            info_dict[self.attribute_name]
            if not isinstance(self.attribute_name, list)
            else th.cat([info_dict[name] for name in self.attribute_name], axis=1)
        )
        obses = data_dict["obs"]
        next_obses = data_dict["next_obs"]
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
            outputs = self.forward(args)
            loss = self.loss_func(outputs, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        return total_loss / num_batches

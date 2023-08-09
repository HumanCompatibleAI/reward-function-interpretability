"""Train probes on reward nets."""

import itertools
import math
import random
from typing import Callable, List, Optional, Tuple, Union
import warnings

from imitation.rewards.reward_nets import CnnRewardNet, cnn_transpose
from stable_baselines3.common import preprocessing
import torch as th
from torch import nn

from reward_preprocessing.common.utils import (
    DoubleInfoTransitionsWithRew,
    copy_module,
    transitions_collate_fn,
)


class CnnProbe(nn.Module):
    # inspired by
    # https://github.com/yukimasano/linear-probes/blob/master/eval_linear_probes.py
    # TODO: add check on attribute_dim
    # TODO: would be nice to reuse reward_net methods to a greater extent than I
    # currently am able to.
    # TODO(df): write documentation
    # TODO(df): make this generic over base reward nets (should be v little work)
    def __init__(
        self,
        reward_net: CnnRewardNet,
        layer_name: str,
        num_probe_layers: int,
        attribute_dim: int,
        attribute_name: Union[str, List[str]],
        attribute_max: Optional[float],
        loss_type: str,
        device: th.device,
        attribute_func: Optional[
            Union[Callable[..., float], Callable[..., List[float]]]
        ] = None,
        obs_shape: Tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        """Make a probe on an underlying CNN.


        num_probe_layers: Number of hidden layers in the probe head.
        attribute_func: Function to call on the attribute in the info dict before
            regressing the probe. Defaults to the identity. Attribute_dim should be the
            length of the output of this function.
        """
        super(CnnProbe, self).__init__()
        self.attribute_name = attribute_name
        self.attribute_dim = attribute_dim
        self.attribute_max = attribute_max
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
        self.num_probe_layers = num_probe_layers
        self.loss_type = loss_type
        self.attribute_func = attribute_func
        self.device = device

        # type-check inputs
        if self.attribute_max is not None and self.attribute_max <= 0:
            raise ValueError(
                f"attribute_max was {self.attribute_max}, should be positive."
            )

        if self.loss_type not in ["mse", "cross_entropy"]:
            raise ValueError(
                f"loss_type was {self.loss_type}, should be mse or cross_entropy"
            )
        elif self.loss_type == "mse":
            self.loss_func = nn.MSELoss()
        else:
            assert self.loss_type == "cross_entropy"
            self.loss_func = nn.CrossEntropyLoss()

        # get shape of reward_net inputs, to find activation shape at layer_name,
        # to determine the shape of the linear layer at the end of the probe.
        input_channels = 0

        if self.use_state:
            input_channels += obs_shape[0]

        if self.use_next_state:
            input_channels += obs_shape[0]

        if not (self.use_state or self.use_next_state):
            raise ValueError("Reward net must use state or next_state")

        self.model.to(self.device)

        x = th.zeros(1, input_channels, obs_shape[1], obs_shape[2]).to(device)
        layer_count = 0
        layers = []
        started_probe = False
        added_probe = False
        # man I wish I had broken out the function that took sas' to a tensor
        # when I was writing CnnRewardNet.
        all_names = []
        for name, child in self.model.named_children():
            # Iterate thru modules of the reward net.
            # Once you've hit self.layer_name, start adding layers to the probe head of
            # the same type as layers of the network.
            # Increment the layer_count for each activation function you encounter in
            # the network, until it hits self.num_probe_layers. At that point, add a
            # linear head to the end of the probe.
            x = child.forward(x)
            if started_probe and not added_probe:
                layers.append(copy_module(child))
                # Checks whether we've hit an activation function
                if isinstance(
                    child, (nn.ReLU, nn.ELU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, nn.Tanh)
                ):
                    layer_count += 1
            all_names.append(name)
            if name == self.layer_name:
                if started_probe:
                    warnings.warn(
                        f"Reward network has multiple layers named {name}, probe will "
                        + "commence at first such layer."
                    )
                started_probe = True
            if (
                started_probe
                and layer_count == self.num_probe_layers
                and not added_probe
            ):
                if len(x.shape) > 2:
                    avg_pool = nn.AdaptiveAvgPool2d(1)
                    flatten = nn.Flatten()
                    layers += [avg_pool, flatten]
                fc = nn.Linear(x.size(1), attribute_dim)
                layers.append(fc)
                self.probe_head = nn.Sequential(*layers)
                added_probe = True

        # This code should work such that if added_probe is True, then started_probe
        # should also be True. The following assert checks that material conditional.
        assert started_probe or not added_probe

        if not started_probe:
            raise ValueError(
                f"Could not find layer {self.layer_name} to probe. "
                + f"List of layer names: {all_names}"
            )

        if layer_count < self.num_probe_layers:
            raise ValueError(
                f"Attempted to make probe with {self.num_probe_layers} layers, which is"
                + " more layers than in the remaining network."
            )

        # if the above errors weren't raised, the layer should have been added.
        assert added_probe

        self.probe_head.to(self.device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # feed input thru the network until hitting the probed layer, where you apply
        # the probe head.
        self.model.eval()
        for name, child in self.model.named_children():
            x = child.forward(x)
            if name == self.layer_name:
                return self.probe_head(x)
        assert False, f"Could not find layer {self.layer_name} to probe."

    def train(
        self,
        dataset: DoubleInfoTransitionsWithRew,
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
        train_loader, test_loader, num_train_batches = self.make_loaders(
            dataset, frac_train, batch_size
        )
        init_train_loss = self.eval_on_dataloader(train_loader)
        print("Initial train loss:", init_train_loss)
        init_test_loss = self.eval_on_dataloader(test_loader)
        print("Initial test loss:", init_test_loss)
        optimizer = th.optim.Adam(self.probe_head.parameters())
        for epoch in range(num_epochs):
            epoch_loss = self.eval_on_dataloader(
                train_loader, optimizer=optimizer, num_train_batches=num_train_batches
            )
            print(
                f"Average train loss over last 10 batches of epoch {epoch}:", epoch_loss
            )
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
        num_train_batches = math.ceil(num_train / batch_size)
        test_loader = th.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=transitions_collate_fn,
        )
        return train_loader, test_loader, num_train_batches

    def data_dict_to_args_and_target(
        self, data_dict: dict
    ) -> Tuple[th.Tensor, th.Tensor]:
        # extract state variables being probed for, and create inputs to the reward net.
        info_dicts = (
            data_dict["next_infos"] if self.use_next_state else data_dict["infos"]
        )
        attr_func = (
            self.attribute_func if self.attribute_func is not None else (lambda x: x)
        )
        target = (
            th.tensor(
                [
                    list(
                        itertools.chain.from_iterable(
                            [
                                self.cap_state_var(attr_func(info_dict[name]))
                                for name in self.attribute_name
                            ]
                        )
                    )
                    for info_dict in info_dicts
                ]
            ).to(th.float32)
            if isinstance(self.attribute_name, list)
            else th.tensor(
                [
                    self.cap_state_var(attr_func(info_dict[self.attribute_name]))
                    for info_dict in info_dicts
                ]
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

    def cap_state_var_(self, state_var):
        if self.attribute_max is not None:
            mined_var = min(state_var, self.attribute_max)
            maxed_var = max(mined_var, -self.attribute_max)
            return maxed_var
        else:
            return state_var

    def cap_state_var(self, state_var):
        if isinstance(state_var, list):
            return list(map(self.cap_state_var_, state_var))
        else:
            return [self.cap_state_var_(state_var)]

    def eval_on_dataloader(
        self,
        data_loader: th.utils.data.DataLoader,
        optimizer: Optional[th.optim.Optimizer] = None,
        num_train_batches: Optional[int] = None,
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
        for i, data in enumerate(data_loader):
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
                if num_train_batches - i < 10:
                    total_loss += loss.item()
                    num_batches += 1
            else:
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

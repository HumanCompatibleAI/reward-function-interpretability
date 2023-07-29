import functools
import operator
import os
import os.path
from typing import List, Optional, Union

from imitation.data import types
from imitation.scripts.common import common, demonstrations
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common.utils import (
    DoubleInfoTransitionsWithRew,
    flatten_trajectories_with_rew_double_info,
)
from reward_preprocessing.models import CnnRewardNetWorkaround
from reward_preprocessing.probes import CnnProbe
from reward_preprocessing.scripts.config.train_probe import train_probe_ex


@train_probe_ex.capture
def train_probe(
    dataset,
    reward_net,
    device,
    layer_name,
    num_probe_layers,
    attributes,
    attr_dim,
    attr_cap,
    batch_size,
    num_epochs,
):
    """Train a probe on the provided reward net and trajectories."""
    probe = CnnProbe(
        reward_net,
        layer_name=layer_name,
        num_probe_layers=num_probe_layers,
        attribute_dim=attr_dim,
        attribute_name=attributes,
        attribute_max=attr_cap,
        loss_type="mse",
        device=device,
    )

    probe.train(
        dataset=dataset,
        frac_train=0.9,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


def _sum_vecs(vec1, vec2):
    return list(map(operator.add, vec1, vec2))


def get_mean_attr_val(list_attr_vals):
    num_vals = len(list_attr_vals)
    if isinstance(list_attr_vals[0], list):
        attr_sum = functools.reduce(_sum_vecs, list_attr_vals[1:], list_attr_vals[0])
        norm_sum = [sum_val / num_vals for sum_val in attr_sum]
        return norm_sum
    else:
        return sum(list_attr_vals) / num_vals


def get_mse(list_attr_vals):
    mean = get_mean_attr_val(list_attr_vals)

    def get_mse_single_val(attr_val):
        if isinstance(list_attr_vals[0], list):
            mse = 0
            for attr_scalar, mean_scalar in zip(attr_val, mean):
                mse += (attr_scalar - mean_scalar) ** 2
            return mse / len(list_attr_vals[0])
        else:
            return (attr_val - mean) ** 2

    return sum(map(get_mse_single_val, list_attr_vals)) / len(list_attr_vals)


@train_probe_ex.capture
def benchmark_accuracy(
    dataset,
    use_next_info: bool,
    attributes: Union[str, List[str]],
    attr_cap: Optional[float],
):
    """Determine the MSE from always guessing the mean value of the attributes."""
    attr_list = attributes if isinstance(attributes, list) else [attributes]
    mse = 0
    for attr in attr_list:
        attr_vals = list(
            map(lambda x: x["next_infos" if use_next_info else "infos"][attr], dataset)
        )
        if attr_cap is not None:
            if attr_cap <= 0:
                raise ValueError("Attribute cap must be positive")
            attr_vals = list(
                map(lambda x: max(min(x, attr_cap), (-1) * attr_cap), attr_vals)
            )
        attr_mse = get_mse(attr_vals)
        mse += attr_mse
    print("\nLoss from predicting mean:", mse)


@train_probe_ex.capture
def filter_dataset(dataset, use_next_info, attributes, attr_cap):
    """Filter out transitions where attribute values are larger than the cap."""
    use_infos = "next_infos" if use_next_info else "infos"
    attr_list = attributes if isinstance(attributes, list) else [attributes]
    keep_indices = [
        i
        for i, trans in enumerate(dataset)
        if all(
            [
                trans[use_infos][attr] < attr_cap
                and (-1) * attr_cap < trans[use_infos][attr]
                for attr in attr_list
            ]
        )
    ]
    new_acts = [dataset.acts[i] for i in keep_indices]
    new_obs = [dataset.obs[i] for i in keep_indices]
    new_next_obs = [dataset.next_obs[i] for i in keep_indices]
    new_dones = [dataset.dones[i] for i in keep_indices]
    new_rews = [dataset.rews[i] for i in keep_indices]
    new_infos = [dataset.infos[i] for i in keep_indices]
    new_next_infos = [dataset.next_infos[i] for i in keep_indices]
    print("Length of filtered dataset:", len(keep_indices))
    return DoubleInfoTransitionsWithRew(
        acts=new_acts,
        obs=new_obs,
        next_obs=new_next_obs,
        dones=new_dones,
        rews=new_rews,
        infos=new_infos,
        next_infos=new_next_infos,
    )


@train_probe_ex.main
def train_probes_experiment(
    supervised,  # from ingredient
    traj_path: str,
    reward_net_path: str,
    layer_name: str,
    num_probe_layers: int,
    attributes: Union[str, List[str]],
    attr_cap: Optional[float],
    batch_size: int,
    num_epochs: int,
    compare_to_mean: bool,
    compare_to_random_net: bool,
    filter_extreme_attrs: bool,
):
    trajs = demonstrations.load_expert_trajs(
        rollout_path=traj_path,
        n_expert_demos=None,
    )
    assert isinstance(trajs[0], types.TrajectoryWithRew)
    dataset = flatten_trajectories_with_rew_double_info(trajs)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    reward_net = th.load(reward_net_path, map_location=device)
    if filter_extreme_attrs:
        dataset = filter_dataset(dataset, use_next_info=reward_net.use_next_state)

    train_probe(dataset, reward_net, device)

    if compare_to_mean:
        benchmark_accuracy(dataset, use_next_info=reward_net.use_next_state)

    if compare_to_random_net:
        print("\nTraining probe on randomly initialized network:")
        with common.make_venv() as venv:
            new_net = CnnRewardNetWorkaround(
                **supervised["net_kwargs"],
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                use_done=False,
            )
            train_probe(dataset, new_net, device)


def main():
    observer = FileStorageObserver(os.path.join("../output", "sacred", "train_probe"))
    train_probe_ex.observers.append(observer)
    train_probe_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

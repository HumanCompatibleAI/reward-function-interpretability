import os
import os.path
from typing import List, Union

from imitation.data import types
from imitation.scripts.common import common, demonstrations
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common.utils import flatten_trajectories_with_rew_double_info
from reward_preprocessing.models import CnnRewardNetWorkaround
from reward_preprocessing.probes import Probe
from reward_preprocessing.scripts.config.train_probe import train_probe_ex


@train_probe_ex.capture
def train_probe(
    dataset,
    reward_net,
    device,
    layer_name,
    attributes,
    attr_dim,
    batch_size,
    num_epochs,
):
    """Train a probe on the provided reward net and trajectories."""
    probe = Probe(
        reward_net,
        layer_name=layer_name,
        attribute_dim=attr_dim,
        attribute_name=attributes,
        loss_type="mse",
        device=device,
    )

    probe.train(
        dataset=dataset,
        frac_train=0.9,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


@train_probe_ex.capture
def benchmark_accuracy(dataset, use_next_info: bool, attributes: Union[str, List[str]]):
    """Determine the MSE from always guessing the mean value of the attributes."""
    attr_list = attributes if isinstance(attributes, list) else [attributes]
    mse = 0
    for attr in attr_list:
        vec = list(
            map(lambda x: x["next_infos" if use_next_info else "infos"][attr], dataset)
        )
        mean = sum(vec) / len(vec)
        attr_mse = sum(map(lambda x: (x - mean) ** 2, vec)) / len(vec)
        mse += attr_mse
    print("\nLoss from predicting mean:", mse)


@train_probe_ex.main
def run_experiment(
    supervised,  # from ingredient
    traj_path: str,
    reward_net_path: str,
    layer_name: str,
    attributes: Union[str, List[str]],
    batch_size: int,
    num_epochs: int,
    compare_to_mean: bool,
    compare_to_random_net: bool,
):
    trajs = demonstrations.load_expert_trajs(
        rollout_path=traj_path,
        n_expert_demos=None,
    )
    assert isinstance(trajs[0], types.TrajectoryWithRew)
    dataset = flatten_trajectories_with_rew_double_info(trajs)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    reward_net = th.load(reward_net_path, map_location=device)

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

import os.path as osp
from typing import Sequence, cast, Dict, Tuple

import sacred
import torch as th
from imitation.data import types
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.scripts.common import common, demonstrations  # reward, rl, train
from sacred.observers import FileStorageObserver
from tqdm import tqdm

from reward_preprocessing.models import ProcgenCnnRegressionRewardNet
from reward_preprocessing.scripts.common import supervised as supervised_fns

train_regression_ex = sacred.Experiment(
    "train_regression",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        supervised_fns.supervised_ingredient,
        # reward.reward_ingredient,
        # rl.rl_ingredient,
        # train.train_ingredient,
    ],
)


@train_regression_ex.main
def train_regression(
        supervised  # From ingredient
):
    # Load expert trajectories
    expert_trajs = demonstrations.load_expert_trajs()
    assert isinstance(expert_trajs[0], types.TrajectoryWithRew)
    expert_trajs = cast(Sequence[types.TrajectoryWithRew], expert_trajs)
    dataset = flatten_trajectories_with_rew(expert_trajs)

    train_loader, test_loader = supervised_fns.train_test_dataloaders(dataset)

    venv = common.make_venv()

    # Init the regression CNN
    model = ProcgenCnnRegressionRewardNet(
        observation_space=venv.observation_space, action_space=venv.action_space
    )
    device = "cuda" if th.cuda.is_available() else "cpu"
    loss_fn = th.nn.MSELoss()

    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    _train_supervised(
        num_epochs=supervised["epochs"],
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        test_freq=supervised["test_freq"],
    )


def _train_supervised(
    num_epochs, model, device, train_loader, optimizer, loss_fn, test_loader, test_freq
):
    for epoch in range(1, num_epochs + 1):
        _train_batch(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            loss_fn,
            test_loader,
            test_freq,
        )


def _train_batch(
    model, device, train_loader, optimizer, epoch, loss_fn, test_loader, test_freq
):
    model.train()
    bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=80)
    for batch_idx, data_dict in enumerate(bar):
        model_args, target = _data_dict_to_model_args_and_target(data_dict, device)

        optimizer.zero_grad()
        output = model(*model_args)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % test_freq == 0:  # Test and log every test_freq batches
            # test_loss = _test(model, device, test_loader, loss_fn)
            description = (
                f"Epoch: {epoch}, train loss: {loss.item():.4f}, "
                # f"test loss: {test_loss:.4f}"
            )
            bar.set_description(description)
    bar.close()


def _test(model, device, test_loader, loss_fn) -> float:
    """Test model on data in test_loader. Returns average batch loss."""
    model.eval()
    test_loss: th.Tensor = th.Tensor([0.0])
    with th.no_grad():
        for data_dict in test_loader:
            model_args, target = _data_dict_to_model_args_and_target(data_dict, device)
            output = model(*model_args)
            test_loss += loss_fn(output, target)  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    model.train()

    return test_loss.item()


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


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_regression"))
    train_regression_ex.observers.append(observer)
    train_regression_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

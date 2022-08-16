import os.path as osp
from typing import Sequence, cast

import sacred
import torch as th
from imitation.data import types
from imitation.scripts.common import common, demonstrations  # reward, rl, train
from sacred.observers import FileStorageObserver
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from reward_preprocessing.models import ProcgenCnnRegressionRewardNet
from reward_preprocessing.supervised.data import ObsRewDataset

train_regression_ex = sacred.Experiment(
    "train_regression",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        # reward.reward_ingredient,
        # rl.rl_ingredient,
        # train.train_ingredient,
    ],
)


@train_regression_ex.config
def defaults():
    epochs = 100
    test_frac = 0.1
    batch_size = 32
    num_loader_workers = 0


@train_regression_ex.main
def train_regression(
    epochs: int, test_frac: float, batch_size: int, num_loader_workers: int
):
    # Load expert trajectories
    expert_trajs = demonstrations.load_expert_trajs()
    assert type(expert_trajs) == Sequence[types.TrajectoryWithRew]
    expert_trajs = cast(Sequence[types.TrajectoryWithRew], expert_trajs)
    dataset = ObsRewDataset(expert_trajs)
    # Calculate train-test split
    num_test = int(len(dataset) * test_frac)
    num_train = len(dataset) - num_test
    train, test = random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_loader_workers,
    )

    venv = common.make_venv()

    # Train the regression CNN
    model = ProcgenCnnRegressionRewardNet(
        observation_space=venv.observation_space, action_space=venv.action_space
    )
    device = "cuda" if th.cuda.is_available() else "cpu"
    loss_fn = th.nn.MSELoss()

    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    _train_supervised(
        num_epochs=epochs,
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
    )


def _train_supervised(num_epochs, model, device, train_loader, optimizer, loss_fn):
    for epoch in range(1, num_epochs + 1):
        _train_batch(model, device, train_loader, optimizer, epoch, loss_fn)


def _train_batch(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=80)
    for batch_idx, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 64 == 0:  # Log every 64 batches
            description = f"Epoch: {epoch}, train loss: {loss.item():.4f}"
            bar.set_description(description)
    bar.close()


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_regression"))
    train_regression_ex.observers.append(observer)
    train_regression_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

import os
import os.path as osp
from typing import Sequence, cast

import sacred
import torch as th
from imitation.data import types
from imitation.scripts.common import common, demonstrations
from sacred.observers import FileStorageObserver

from reward_preprocessing.models import ProcgenCnnRegressionRewardNet
from reward_preprocessing.scripts.common import supervised as supervised_config
from reward_preprocessing.trainers.supervised import SupervisedTrainer

train_regression_ex = sacred.Experiment(
    "train_regression",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        supervised_config.supervised_ingredient,
    ],
)


@train_regression_ex.config
def defaults():
    # Every checkpoint_epoch_interval epochs, save the model. Epochs start at 1.
    checkpoint_epoch_interval = 1


def save(trainer: SupervisedTrainer, save_path):
    """Save regression model."""
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_net, os.path.join(save_path, "model.pt"))


@train_regression_ex.main
def train_regression(supervised, checkpoint_epoch_interval: int):  # From ingredient
    # TODO: make function return some stats
    # Load expert trajectories
    expert_trajs = demonstrations.load_expert_trajs()
    assert isinstance(expert_trajs[0], types.TrajectoryWithRew)
    expert_trajs = cast(Sequence[types.TrajectoryWithRew], expert_trajs)

    custom_logger, log_dir = common.setup_logging()

    venv = common.make_venv()

    # Init the regression CNN
    model = ProcgenCnnRegressionRewardNet(
        observation_space=venv.observation_space, action_space=venv.action_space
    )
    device = "cuda" if th.cuda.is_available() else "cpu"
    loss_fn = th.nn.MSELoss()

    trainer = SupervisedTrainer(
        demonstrations=expert_trajs,
        reward_net=model,
        batch_size=supervised["batch_size"],
        test_frac=supervised["test_frac"],
        test_freq=supervised["test_freq"],
        num_loader_workers=supervised["num_loader_workers"],
        loss_fn=loss_fn,
        opt_kwargs={"lr": 1e-3},
        custom_logger=custom_logger,
        allow_variable_horizon=True,
    )

    def checkpoint_callback(epoch_num):
        if checkpoint_epoch_interval > 0 and epoch_num % checkpoint_epoch_interval == 0:
            save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch_num:05d}"))

    # Start training
    trainer.train(
        num_epochs=supervised["epochs"],
        device=device,
        callback=checkpoint_callback,
    )

    # Save final artifacts.
    if checkpoint_epoch_interval >= 0:
        save(trainer, os.path.join(log_dir, "checkpoints", "final"))


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_regression"))
    train_regression_ex.observers.append(observer)
    train_regression_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

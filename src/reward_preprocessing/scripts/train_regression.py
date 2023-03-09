import os
import os.path
from typing import Sequence, cast

from imitation.data import types
from imitation.scripts.common import common, demonstrations
from sacred.observers import FileStorageObserver
import torch as th

import reward_preprocessing.scripts.common.supervised as supervised_config
from reward_preprocessing.models import CnnRewardNetWorkaround
from reward_preprocessing.scripts.config.train_regression import train_regression_ex
from reward_preprocessing.trainers.supervised_trainer import SupervisedTrainer


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

    with common.make_venv() as venv:
        # Init the regression CNN
        model = CnnRewardNetWorkaround(
            **supervised["net_kwargs"],
            # We don't want the following to be overriden.
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            use_done=False,
        )
    _log_model_info(custom_logger, model)

    device = "cuda" if th.cuda.is_available() else "cpu"

    # Move model to correct device
    model.to(device)

    trainer = supervised_config.make_trainer(
        expert_trajectories=expert_trajs, model=model, custom_logger=custom_logger
    )

    trainer.log_data_stats()

    # Log samples
    if supervised["debugging"]["show_samples"]:
        trainer.log_samples(log_as_step=supervised["debugging"]["show_samples_as_step"])

    def checkpoint_callback(epoch_num):
        if checkpoint_epoch_interval > 0 and epoch_num % checkpoint_epoch_interval == 0:
            save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch_num:05d}"))

    custom_logger.log("Start training regression model.")
    # Start training
    trainer.train(
        num_epochs=supervised["epochs"],
        device=device,
        callback=checkpoint_callback,
    )

    # Save final artifacts.
    if checkpoint_epoch_interval >= 0:
        save(trainer, os.path.join(log_dir, "checkpoints", "final"))


def _log_model_info(custom_logger, model):
    custom_logger.log(model)
    if isinstance(model, CnnRewardNetWorkaround):
        # These do not exist in all reward nets. However, they exist in CnnRewardNet.
        custom_logger.log(f"use_state: {model.use_state}")
        custom_logger.log(f"use_action: {model.use_action}")
        custom_logger.log(f"use_next_state: {model.use_next_state}")


def main():
    observer = FileStorageObserver(
        os.path.join("../output", "sacred", "train_regression")
    )
    train_regression_ex.observers.append(observer)
    train_regression_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

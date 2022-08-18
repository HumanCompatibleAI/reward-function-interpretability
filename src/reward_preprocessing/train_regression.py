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
from reward_preprocessing.scripts.common import supervised as supervised_config
from reward_preprocessing.trainers.supervised import SupervisedTrainer

train_regression_ex = sacred.Experiment(
    "train_regression",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        supervised_config.supervised_ingredient,
        # reward.reward_ingredient,
        # rl.rl_ingredient,
        # train.train_ingredient,
    ],
)


@train_regression_ex.main
def train_regression(supervised):  # From ingredient
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

    # Start training
    trainer.train(num_epochs=supervised["epochs"], device=device)


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_regression"))
    train_regression_ex.observers.append(observer)
    train_regression_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

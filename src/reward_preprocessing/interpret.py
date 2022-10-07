import os.path as osp
from typing import Optional, Sequence, cast

import matplotlib
import wandb

matplotlib.use("TkAgg")
from imitation.data import types
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations
from lucent.misc.io import show
from matplotlib import pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common.serialize import load_reward
from reward_preprocessing.vis.reward_vis import LayerNMF

interpret_ex = Experiment(
    "interpret",
    ingredients=[common_config.common_ingredient]
    # ingredients=[demonstrations.demonstrations_ingredient],
)


@interpret_ex.config
def defaults():
    # Path to the learned supervised reward net
    reward_path = None
    # Rollouts to use vor dataset visualization
    rollout_path = None
    n_expert_demos = None
    limit_num_obs = -1

    locals()  # quieten flake8


@interpret_ex.main
def interpret(
    common: dict,  # from sacred config
    reward_path: Optional[str],
    rollout_path: str,
    n_expert_demos: Optional[int],
    limit_num_obs: int,
):
    """Sanity check a learned supervised reward net. Evaluate 4 things:
    - Random policy on env reward
    - Random policy on learned reward function
    - Expert policy on env reward
    - Expert policy on learned reward function
    """
    # Load reward not pytorch module
    if th.cuda.is_available():
        rew_net = th.load(str(reward_path))  # Load from same device as saved
    else:  # CUDA not available
        rew_net = th.load(str(reward_path), map_location=th.device("cpu"))  # Force CPU

    # Set up imitation-style logging
    custom_logger, log_dir = common_config.setup_logging()

    wandb_logging = 'wandb' in common['log_format_strs']

    rew_net.eval()
    # Argument venv not necessary, as it is ignored for SupvervisedRewardNet
    # rew_fn = load_reward("SupervisedRewardNet", reward_path, venv=None)
    # trajs = types.load(rollout_path)

    # Load trajectories for dataset visualization
    expert_trajs = demonstrations.load_expert_trajs(rollout_path, n_expert_demos)
    assert isinstance(expert_trajs[0], types.TrajectoryWithRew)
    expert_trajs = cast(Sequence[types.TrajectoryWithRew], expert_trajs)
    from lucent.modelzoo.util import get_model_layers

    print("Available layers:")
    print(get_model_layers(rew_net))

    # Get observations from trajectories
    observations = np.concatenate([traj.obs for traj in expert_trajs])
    
    if limit_num_obs < 0:
        obses = observations
    else:
        obses = observations[:limit_num_obs]
    nmf = LayerNMF(
        model=rew_net,
        features=2,
        layer_name="cnn_regressor_dense_final",
        # layer_name="cnn_regressor_avg_pool",
        obses=obses,
        activation_fn="sigmoid",
    )

    # Visualization
    num_features = nmf.features
    rows, columns = 1, num_features
    fig = plt.figure(figsize=(columns * 2, rows * 2))  # width, height in inches
    for i in range(num_features):
        print(i)

        img, indices = nmf.vis_dataset_thumbnail(feature=i, num_mult=4, expand_mult=1)
        # img = img.astype(np.uint8)
        # index = indices[0][0]
        # img = observations[index]

        if wandb_logging:
            wb_img = wandb.Image(img, caption=f"Feature {i}")
            custom_logger.record(f"feature_{i}", wb_img)

        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
        # show()
    if wandb_logging:
        custom_logger.dump(step=0)

    plt.show()


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

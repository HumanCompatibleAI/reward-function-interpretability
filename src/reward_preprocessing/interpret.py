import os.path as osp
from typing import Optional, Sequence, cast
import matplotlib
matplotlib.use('TkAgg')
from imitation.data import types
from imitation.scripts.common import demonstrations
import numpy as np
from lucent.misc.io import show
from matplotlib import pyplot as plt
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common.serialize import load_reward
from reward_preprocessing.vis.reward_vis import LayerNMF

interpret_ex = Experiment(
    "interpret",
    # ingredients=[demonstrations.demonstrations_ingredient],
)


@interpret_ex.config
def defaults():
    # Path to the learned supervised reward net
    reward_path = None
    # Rollouts to use vor dataset visualization
    rollout_path = None
    n_expert_demos = None

    locals()  # quieten flake8


@interpret_ex.main
def interpret(
    reward_path: Optional[str],
    rollout_path: str,
    n_expert_demos: Optional[int],
):
    """Sanity check a learned supervised reward net. Evaluate 4 things:
    - Random policy on env reward
    - Random policy on learned reward function
    - Expert policy on env reward
    - Expert policy on learned reward function
    """
    if th.cuda.is_available():
        rew_net = th.load(str(reward_path))  # Load from same device as saved
    else:  # CUDA not available
        rew_net = th.load(str(reward_path), map_location=th.device("cpu"))  # Force CPU

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

    nmf = LayerNMF(
        model=rew_net,
        # layer_name="cnn_regressor_dense_final",
        layer_name="cnn_regressor_avg_pool",
        obses=observations[:1024],
    )
    for i in range(nmf.features):
        print(i)

        plt.figure()
        img, indices = nmf.vis_dataset(feature=i, expand_mult=1)
        # img = img.astype(np.uint8)
        index = indices[0][0]
        img = observations[index]
        plt.imshow(img)
        # show()
        plt.show()


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

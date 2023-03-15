from typing import Sequence, cast

from imitation.data import types
from imitation.data.rollout import flatten_trajectories_with_rew
from imitation.scripts.common import demonstrations


def benchmark_mse(traj_path):
    """Test the mean squared error of always predicting the mean reward of this dataset.

    traj_path: str, path to a file containing a sequence of `types.Trajectory` saved in
        numpy's npz format.
    """
    expert_trajs = demonstrations.load_expert_trajs(
        rollout_path=traj_path, n_expert_demos=None
    )
    assert isinstance(expert_trajs[0], types.TrajectoryWithRew)
    expert_trajs = cast(Sequence[types.TrajectoryWithRew], expert_trajs)
    dataset = flatten_trajectories_with_rew(expert_trajs)
    rews = list(map(lambda x: x["rews"], dataset))
    mean_rew = sum(rews) / len(rews)
    print(f"Mean reward: {mean_rew}")
    mse = sum(map(lambda x: (x - mean_rew) ** 2, rews)) / len(rews)
    print(f"MSE from always guessing the mean: {mse}")


if __name__ == "__main__":
    path = (
        "/home/daniel/reward-function-interpretability/"
        + "bigfish_rollouts_3e4_episodes_2023-02.npz"
    )
    benchmark_mse(path)

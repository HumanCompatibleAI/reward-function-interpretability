from typing import Sequence, Tuple

import imitation.data.types as types
import numpy as np
from torch.utils.data import Dataset


class ObsRewDataset(Dataset):
    """Dataset iterating over the observations and rewards of a sequence of trajectories
    (with reward)."""

    def __init__(self, trajectories: Sequence[types.TrajectoryWithRew]):
        self.trajectories = trajectories

    def __len__(self):
        length = 0
        for trajectory in self.trajectories:
            length += len(trajectory)
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        original_idx = idx
        for traj in self.trajectories:
            if idx < len(traj):
                obs: np.ndarray = traj.obs[idx]
                rew: float = traj.rews[idx]
                return obs, rew
            idx -= len(traj)
        raise ValueError(f"idx out of range: idx: {original_idx}, len: {len(self)}")

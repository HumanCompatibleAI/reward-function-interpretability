import os

import numpy as np

DATA_PATH = (
    "/nas/ucb/daniel/nas_reward_function_interpretability/"
    + "dots-and-dists-64-1e6-2023-11.npz"
)
SAVE_PATH = (
    "/nas/ucb/daniel/nas_reward_function_interpretability/"
    + "dots-and-dists-64-1e6-2023-11-binarized.npz"
)
NON_ZERO_FRAC = 0.0093

traj_data = np.load(DATA_PATH, allow_pickle=True)

rews_sorted = sorted(traj_data["rews"])
low_avg_dist = rews_sorted[int(NON_ZERO_FRAC * len(rews_sorted))]
new_rews = list(map(lambda rew: 10.0 if rew < low_avg_dist else 0.0, traj_data["rews"]))

new_traj_data = {
    "obs": traj_data["obs"],
    "acts": traj_data["acts"],
    "infos": traj_data["infos"],
    "terminal": traj_data["terminal"],
    "rews": np.array(new_rews).astype(np.float32),
    "indices": traj_data["indices"],
}

tmp_path = SAVE_PATH + ".tmp"
with open(tmp_path, "wb") as f:
    np.savez_compressed(f, **new_traj_data)

os.replace(tmp_path, SAVE_PATH)
print("Saved binarized trajectory")

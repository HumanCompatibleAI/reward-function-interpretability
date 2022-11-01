"""Fix saved trajectory format from that one time that I saved them wrong."""
import numpy as np

path = "/home/pavel/out/interpret/expert-rollouts/procgen-gm/005/fixed-coin_1000.2k.npz"
data = np.load(path, allow_pickle=True)

# Observations need to be fixed
observations = data["obs"]

indices = data["indices"]
traj_list = []
for i in range(len(indices)):
    if i == 0:
        start = 0
    else:
        start = indices[i - 1]
    end = indices[i]
    # + 1 because we also want to include the last next_obs
    obs = observations[start : end + 1]
    traj_list.append(obs)
# Also add the last trajectory
traj_list.append(observations[indices[-1] :])

# Concatenate them together, duplicates and all
new_observations = np.concatenate(traj_list, axis=0)

# Sanity check
assert (
    np.cumsum([len(traj) - 1 for traj in traj_list[:-1]]) == np.array(indices)
).all()

new_dict = {
    "obs": new_observations,
    "acts": data["acts"],
    "infos": data["infos"],
    "terminal": data["terminal"],
    "rews": data["rews"],
    "indices": data["indices"],
}

# Update path name
split = path.split(".")
split[-2] += "_fixed"
save_path = ".".join(split)

# Save fixed data
with open(save_path, "wb") as f:
    np.savez_compressed(f, **new_dict)

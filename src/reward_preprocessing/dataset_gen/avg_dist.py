import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

size = (2, 2)  # In inches
number_pairs = 3
circle_radius = 0.05
colors = ["r", "b", "g"]  # Add more colores to allow more pairs
seed = 0
num_transitions = 200
traj_path = "/nas/ucb/pavel/rfi/test/simple-env-traj.npz"

# Set the seed
np.random.seed(seed)

if number_pairs > len(colors):
    raise ValueError("Not enough colors for the number of pairs")


def generate_transition():
    fig, ax = plt.subplots()
    fig.set_size_inches(size)

    def random_coordinate():
        return np.random.uniform(0 + circle_radius, 1 - circle_radius)

    distances = []  # Collect distances between same-colored circles
    for pair_i in range(number_pairs):
        a_x = random_coordinate()
        a_y = random_coordinate()
        a = plt.Circle(
            (a_x, a_y),
            circle_radius,
            color=colors[pair_i],
            clip_on=False,
        )

        ax.add_patch(a)

        b_x = random_coordinate()
        b_y = random_coordinate()
        b = plt.Circle(
            (b_x, b_y),
            circle_radius,
            color=colors[pair_i],
            clip_on=False,
        )

        ax.add_patch(b)

        distance = np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
        distances.append(distance)
    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    # fig.savefig("plotcircles.png", bbox_inches="tight")
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    avg_distance = np.mean(distances)
    return data, avg_distance, distances


obs_list = []
infos_list = []
avg_distances = []
for i in range(num_transitions):
    data, avg_distance, distances = generate_transition()
    obs_list.append(data)
    infos_list.append({"distances": distances})
    avg_distances.append(avg_distance)

# Duplicate last observation, since there is always a final next_obs.
obs_list.append(obs_list[-1].copy())

condensed = {
    "obs": np.array(obs_list),
    "acts": np.zeros((num_transitions,)).astype(np.int8),
    "infos": np.array(infos_list),
    "terminal": np.array([True] * num_transitions),
    "rews": np.array(avg_distances),
    # The indices are pretty arbitrary, they show where an episode ends. Since we
    # don't have a real RL environment, they are not really meaningful. As they are
    # nevertheless required to save a rollout dataset, we simply choose the index such
    # that there are always 1 episodes in the dataset.
    "indices": np.array([]),
}

# This just adjusts the file name to include the number of timesteps (in 1000s).
save_path = traj_path
tmp_path = save_path + ".tmp"
with open(tmp_path, "wb") as f:
    np.savez_compressed(f, **condensed)

os.replace(tmp_path, save_path)
print(f"Saved trajectories to {save_path}.")

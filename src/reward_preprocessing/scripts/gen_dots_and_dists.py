import os
import os.path
from typing import List, Optional, Tuple

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred.observers import FileStorageObserver

from reward_preprocessing.scripts.config.gen_dots_and_dists import (
    generate_simple_trajectories_ex,
)

matplotlib.use("Agg")


@generate_simple_trajectories_ex.main
def generate_simple_trajectories(
    number_pairs: int,
    circle_radius: float,
    num_transitions: int,
    seed: int,
    traj_path: str,
    colors: List[str],
    size: Tuple[int, int],
    weights: Optional[List[int]] = None,
):

    # Set the seed
    np.random.seed(seed)
    obs_list = []
    infos_list = []
    avg_distances = []
    for i in range(num_transitions + 1):
        data, avg_distance, distances = generate_transition(
            number_pairs, circle_radius, size, colors, weights
        )
        obs_list.append(data)
        infos_list.append({"distances": distances})
        avg_distances.append(avg_distance)

    # Drop the first element of the infos list, since the first observation shouldn't
    # come with an info dict (see flatten_trajectories_with_rew_double_info in
    # common/utils.py)
    infos_list = infos_list[1:]

    # Drop the last element of avg_distances, since the last observation is the next_obs
    # of the final transition, and rewards are associated with obs, not next_obs
    avg_distances = avg_distances[:-1]

    condensed = {
        "obs": np.array(obs_list).astype(np.uint8),
        "acts": np.zeros((num_transitions,)).astype(np.int8),
        "infos": np.array(infos_list),
        "terminal": np.array([True] * num_transitions),
        "rews": np.array(avg_distances).astype(np.float32),
        # The indices are pretty arbitrary, they show where an episode ends. Since we
        # don't have a real RL environment, they are not really meaningful. As they are
        # nevertheless required to save a rollout dataset, we simply choose the index
        # such that there is always 1 episode in the dataset.
        "indices": np.array([]),
    }

    save_path = traj_path
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **condensed)

    os.replace(tmp_path, save_path)
    print(f"Saved trajectories to {save_path}.")


def generate_transition(
    number_pairs: int,
    circle_radius: float,
    size: Tuple[float, float],
    colors: List[str],
    weights: Optional[List[int]],
):
    if weights is not None:
        if len(weights) < number_pairs:
            raise ValueError("Not every pair has a weight")
        norm = sum(weights[:number_pairs])

    if number_pairs > len(colors):
        raise ValueError("Not enough colors for the number of pairs")

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

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    if weights is None:
        avg_distance = np.mean(distances)
    else:
        weighted_distances = [
            dist * weight for (dist, weight) in zip(distances, weights)
        ]
        avg_distance = sum(weighted_distances) / norm
    return data, avg_distance, distances


def main():
    observer = FileStorageObserver(
        os.path.join("../output", "sacred", "generate_simple_trajectories")
    )
    generate_simple_trajectories_ex.observers.append(observer)
    generate_simple_trajectories_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

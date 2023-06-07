from imitation.scripts.common import common
import sacred

generate_simple_trajectories_ex = sacred.Experiment(
    "generate_simple_trajectories",
    ingredients=[
        common.common_ingredient,
    ],
)


@generate_simple_trajectories_ex.config
def defaults():
    number_pairs = 3
    circle_radius = 0.05
    num_transitions = 10000
    seed = 0
    traj_path = "/nas/ucb/pavel/rfi/test/dots-and-dists-10000.npz"
    size = (0.64, 0.64)  # Size of the observation in inches
    colors = ["r", "b", "g", "y", "c"]  # Add more colors to allow more pairs
    locals()  # make flake8 happy

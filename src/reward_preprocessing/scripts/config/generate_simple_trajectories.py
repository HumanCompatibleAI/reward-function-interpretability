from imitation.scripts.common import common
import matplotlib
import sacred

matplotlib.use("Agg")

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
    traj_path = "/nas/ucb/pavel/rfi/test/simple-env-10000.npz"
    size = (2, 2)  # Size of the observation in inches
    colors = ["r", "b", "g", "y", "c"]  # Add more colores to allow more pairs
    locals()  # make flake8 happy

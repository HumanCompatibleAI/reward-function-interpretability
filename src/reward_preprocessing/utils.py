import importlib
from pathlib import Path
import tempfile
from typing import Callable, List

import matplotlib.pyplot as plt
import sacred
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs


class ContinuousVideoRecorder(VecVideoRecorder):
    """Modification of the VecVideoRecorder that doesn't restart
    the video when an episode ends.
    """

    def reset(self, start_video=False) -> VecEnvObs:
        obs = self.venv.reset()
        if start_video:
            self.start_video_recorder()
        return obs


class ComposeTransforms:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for trafo in self.transforms:
            x = trafo(x)
        return x


def add_observers(ex: sacred.Experiment) -> None:
    """Add a config hook to a Sacred Experiment which will add configurable observers.

    A 'run_dir' config field must exist for the Experiment.
    """

    def helper(config, command_name, logger):
        # Just to be safe, we check whether an observer already exists,
        # to avoid adding multiple copies of the same observer
        # (see https://github.com/IDSIA/sacred/issues/300)
        if len(ex.observers) == 0:
            ex.observers.append(sacred.observers.FileStorageObserver(config["run_dir"]))

    ex.config_hook(helper)


def sacred_save_fig(fig: plt.Figure, run, filename: str) -> None:
    """Save a matplotlib figure as a Sacred artifact.

    Args:
        fig (plt.Figure): the Figure to be saved
        run: the Sacred run instance (can be obtained via _run
            in captured functions)
        filename (str): the filename for the figure (without extension).
            May also consist of folders, then this hierarchy
            will be respected in the run directory for the Experiment.
    """
    with tempfile.TemporaryDirectory() as dirname:
        plot_path = Path(dirname) / f"{filename}.pdf"
        fig.savefig(plot_path)
        run.add_artifact(plot_path)


def instantiate(module_name: str, class_name: str, **kwargs):
    """Instantiate a class from a string.

    Args:
        module_name (str): the full name of the module the class is located in
        class_name (str): the name of the class to instantiate
        **kwargs: kwargs to pass on to the class constructor

    Returns:
        an instance of the given class
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def get_env_name(env: VecEnv) -> str:
    """Return the name of a vectorized environment (such as 'MountainCar-v0')."""
    # It's only one line but it's a somewhat hard to read and write one
    specs = env.get_attr('spec', indices=[0])
    return specs[0].id

import os.path as osp
from typing import Any, Optional, Sequence, cast

from PIL import Image
from lucent.optvis import transform
import matplotlib
import wandb

from reward_preprocessing.common.networks import (
    ChannelsFirstToChannelsLast,
    FourDimOutput,
    NextStateOnlyModel,
)

from imitation.data import types
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations
from matplotlib import pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.vis.reward_vis import LayerNMF

interpret_ex = Experiment(
    "interpret",
    ingredients=[common_config.common_ingredient]
    # ingredients=[demonstrations.demonstrations_ingredient],
)


@interpret_ex.config
def defaults():
    # Path to the learned supervised reward net
    reward_path = None
    # Rollouts to use vor dataset visualization
    rollout_path = None
    n_expert_demos = None
    # Limit the number of observations to use for visualization, -1 for all
    limit_num_obs = -1
    pyplot = False  # Plot images as pyplot figures
    vis_scale = 4  # Scale the visualization img by this factor
    vis_type = "traditional"  # "traditional" or "dataset"
    layer_name = "reshaped_out"  # Name of the layer to visualize.
    num_features = 2  # Number of features to use for visualization.

    locals()  # quieten flake8


def uncurry_pad_i2_of_4(arg: Any) -> tuple[None, None, Any, None]:
    """Pads output with None such that input arg is at index 2 in the output 4-tuple.
    arg -> (None, None, arg, None)"""
    tuple = (None, None, arg, None)
    return tuple


@interpret_ex.main
def interpret(
    common: dict,  # from sacred config
    reward_path: Optional[str],
    rollout_path: str,
    n_expert_demos: Optional[int],
    limit_num_obs: int,
    pyplot: bool,
    vis_scale: int,
    vis_type: str,
    layer_name: str,
    num_features: int,
):
    """Sanity check a learned supervised reward net. Evaluate 4 things:
    - Random policy on env reward
    - Random policy on learned reward function
    - Expert policy on env reward
    - Expert policy on learned reward function
    """
    # Load reward not pytorch module
    if th.cuda.is_available():
        rew_net = th.load(str(reward_path))  # Load from same device as saved
    else:  # CUDA not available
        rew_net = th.load(str(reward_path), map_location=th.device("cpu"))  # Force CPU

    # Set up imitation-style logging
    custom_logger, log_dir = common_config.setup_logging()

    wandb_logging = "wandb" in common["log_format_strs"]

    rew_net.eval()

    # See description of class for explanation
    rew_net = NextStateOnlyModel(rew_net)

    rew_net = ChannelsFirstToChannelsLast(rew_net)

    # This is due to how lucent works
    # TODO: this should probably be unified instead of having many different exceptions
    if vis_type == "traditional":
        rew_net = FourDimOutput(rew_net)
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
    # Transpose all observations because lucent expects channels first (after batch dim)
    observations = np.transpose(observations, (0, 3, 1, 2))

    if limit_num_obs < 0:
        obses = observations
    else:
        custom_logger.log(
            f"Limiting number of observations to {limit_num_obs} of "
            f"{len(observations)} total."
        )
        obses = observations[:limit_num_obs]
    nmf = LayerNMF(
        model=rew_net,
        features=num_features,
        layer_name=layer_name,
        # layer_name="cnn_regressor_avg_pool",
        obses=obses,
        activation_fn="sigmoid",
    )

    # Visualization
    num_features = nmf.features
    rows, columns = 1, num_features
    if pyplot:
        fig = plt.figure(figsize=(columns * 2, rows * 2))  # width, height in inches
    for i in range(num_features):
        print(i)

        # Get the visualization as image
        if vis_type == "traditional":
            # List of transforms
            transforms = [
                transform.jitter(2),  # Jitters input by 2 pixel
                # Input into model should be 4 tuple, where next_state (3rd arg) is the
                # observation and other inputs are ignored.
                # uncurry_pad_i2_of_4,
            ]

            img = nmf.vis_traditional(transforms=transforms)
        elif vis_type == "dataset":
            img, indices = nmf.vis_dataset_thumbnail(
                feature=i, num_mult=4, expand_mult=1
            )
        else:
            raise ValueError(f"Unknown vis_type: {vis_type}.")
        # img = img.astype(np.uint8)
        # index = indices[0][0]
        # img = observations[index]

        if wandb_logging:
            p_img = Image.fromarray(np.uint8(img * 255), mode="RGBA").resize(
                size=(img.shape[0] * vis_scale, img.shape[1] * vis_scale),
                resample=Image.NEAREST,
            )
            wb_img = wandb.Image(p_img, caption=f"Feature {i}")
            custom_logger.record(f"feature_{i}", wb_img)
        if pyplot:
            if len(img.shape) == 3:
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img)
            elif len(img.shape) == 4:
                for img_i in range(img.shape[0]):
                    fig.add_subplot(rows, columns, i + 1)
                    plt.imshow(img[img_i])

        # show()
    if wandb_logging:
        custom_logger.dump(step=0)
    if pyplot:
        plt.show()
    custom_logger.log("Done with dataset visualization.")


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

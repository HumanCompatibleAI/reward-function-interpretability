import os.path as osp
from typing import Any, Optional, Sequence, cast

from PIL import Image
from imitation.data import types
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations
from imitation.util import logger as imit_logger
from lucent.optvis import transform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th
import wandb

from reward_preprocessing.common.networks import (
    ChannelsFirstToChannelsLast,
    FourDimOutput,
    NextStateOnlyModel,
)
from reward_preprocessing.generative_modelling.utils import tensor_to_transition
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
    gan_path = None
    img_save_path = None

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
    gan_path: Optional[str] = None,
    img_save_path: Optional[str] = None,
):
    """Sanity check a learned supervised reward net. Evaluate 4 things:
    - Random policy on env reward
    - Random policy on learned reward function
    - Expert policy on env reward
    - Expert policy on learned reward function

    img_save_path must be a directory, end in a /.
    """

    if vis_type not in ["dataset", "traditional"]:
        raise ValueError(f"Unknown vis_type: {vis_type}")
    if vis_type == "dataset" and gan_path is not None:
        raise ValueError("GANs cannot be used with dataset visualization.")

    if pyplot:
        matplotlib.use("TkAgg")

    # Load reward not pytorch module

    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    rew_net = th.load(str(reward_path), map_location=device)

    # If GAN path is specified, combine with a GAN.
    if gan_path is not None:
        gan = th.load(gan_path, map_location=device)
        rew_net = RewardGeneratorCombo(reward_net=rew_net, generator=gan.generator)

    # Set up imitation-style logging
    custom_logger, log_dir = common_config.setup_logging()

    wandb_logging = "wandb" in common["log_format_strs"]

    rew_net.eval()

    if vis_type == "traditional" and gan_path is None:
        rew_net = rew_net.cnn_regressor
    elif vis_type == "dataset":
        # See description of class for explanation
        rew_net = NextStateOnlyModel(rew_net)

    # rew_net = ChannelsFirstToChannelsLast(rew_net)

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

    # vis traditional -> channel first,
    # vid dataset -> channel last
    if vis_type == "traditional":
        # Transpose all observations because lucent expects channels first (after
        # batch dim)
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
        device=device,
    )

    # Visualization
    num_features = nmf.features
    rows, columns = 1, num_features
    if pyplot:
        fig = plt.figure(figsize=(columns * 4, rows * 2))  # width, height in inches
    for i in range(num_features):
        print(i)

        # Get the visualization as image
        if vis_type == "traditional" and gan_path is None:
            # List of transforms
            transforms = [
                transform.jitter(2),  # Jitters input by 2 pixel
                # Input into model should be 4 tuple, where next_state (3rd arg) is the
                # observation and other inputs are ignored.
                # uncurry_pad_i2_of_4,
            ]

            next_obs = nmf.vis_traditional(transforms=transforms)
            obs = next_obs
        elif vis_type == "traditional" and gan_path is not None:
            # TODO(df): see if "input" is a legit name.
            latent = nmf.vis_traditional(l2_coeff=0.1, l2_layer_name="input")
            latent_th = th.from_numpy(latent).to(device)
            trans_tens = gan.generator(latent_th)
            obs_th, _, next_obs_th = tensor_to_transition(trans_tens)
            obs = obs_th.detach().cpu().numpy()
            next_obs = next_obs_th.detach().cpu().numpy()
        elif vis_type == "dataset":
            next_obs, indices = nmf.vis_dataset_thumbnail(
                feature=i, num_mult=4, expand_mult=1
            )
            obs = next_obs
        # img = img.astype(np.uint8)
        # index = indices[0][0]
        # img = observations[index]

        if wandb_logging:
            log_arr_to_wandb(
                obs, vis_scale, feature=i, img_type="obs", logger=custom_logger
            )
            log_arr_to_wandb(
                next_obs,
                vis_scale,
                feature=i,
                img_type="next_obs",
                logger=custom_logger,
            )
            # Can't re-use steps unfortunately, so each feature img gets its own step.
            custom_logger.dump(step=i)
        if img_save_path is not None:
            if img_save_path[-1] != "/":
                raise ValueError("img_save_path is not a directory, does not end in /")
            obs_img = array_to_image(obs, vis_scale)
            obs_img.save(img_save_path + f"{i}_obs.png")
            next_obs_img = array_to_image(next_obs, vis_scale)
            next_obs_img.save(img_save_path + f"{i}_next_obs.png")
        if pyplot:
            add_to_figure(fig, obs, "obs")
            add_to_figure(fig, next_obs, "next_obs")

        # show()
    if pyplot:
        plt.show()
    custom_logger.log("Done with dataset visualization.")


def array_to_image(arr: np.ndarray, scale: int) -> Image:
    """Take numpy array on [0,1] scale, return PIL image."""
    return Image.fromarray(np.uint8(arr * 255), mode="RGBA").resize(
        size=(arr.shape[0] * vis_scale, arr.shape[1] * vis_scale),
        resample=Image.NEAREST,
    )


def log_arr_to_wandb(
    arr: np.ndarray,
    scale: int,
    feature: int,
    img_type: str,
    logger: imit_logger.HierarchicalLogger,
) -> None:
    """Log visualized np.ndarray to wandb using given logger.

    Args:
        - arr: array to turn into image, save.
        - scale: ratio by which to scale up the image.
        - feature: which number feature is being visualized.
        - img_type: "obs" or "next_obs"
        - logger: logger to use.
    """
    if img_type not in ["obs", "next_obs"]:
        err_str = f"img_type should be 'obs' or 'next_obs', but instead is {img_type}"
        raise ValueError(err_str)

    pil_img = array_to_image(arr, scale)
    wb_img = wandb.Image(pil_img, caption=f"Feature {feature}, {img_type}")
    logger.record(f"feature_{feature}_{img_type}", wb_img)


def add_to_figure(fig, img: np.ndarray, img_type: str) -> None:
    """Add to pyplot figure"""
    if img_type not in ["obs", "next_obs"]:
        err_str = f"img_type should be 'obs' or 'next_obs', but instead is {img_type}"
        raise ValueError(err_str)

    offset = 1 if img_type == "obs" else 2

    if len(img.shape) == 3:
        fig.add_subplot(rows, columns, 2 * i + offset)
        plt.imshow(img)
    elif len(img.shape) == 4:
        for img_i in range(img.shape[0]):
            fig.add_subplot(rows, columns, 2 * i + offset)
            plt.imshow(img[img_i])
    else:
        raise ValueError("img should have either 3 or 4 dimensions.")


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

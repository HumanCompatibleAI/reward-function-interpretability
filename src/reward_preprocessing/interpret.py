import os.path as osp
from typing import Any, Optional, Sequence, cast

from PIL import Image
from imitation.data import types
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import transform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th
import wandb

from reward_preprocessing.common.networks import FourDimOutput, NextStateOnlyModel
from reward_preprocessing.common.utils import (
    TensorTransitionModel,
    rollouts_to_dataloader,
)
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
    # Limit the number of observations to use for dim reduction.
    # The RL Vision paper uses "a few thousand" observations.
    limit_num_obs = 2048
    pyplot = False  # Plot images as pyplot figures
    vis_scale = 4  # Scale the visualization img by this factor
    vis_type = "traditional"  # "traditional" (gradient-based) or "dataset"
    # Name of the layer to visualize. To figure this out run interpret and the
    # available layers will be printed. For additional notes see interpret doc comment.
    layer_name = "reshaped_out"
    num_features = 2  # Number of features to use for visualization.
    # Path to the GAN model. If None simpy visualize reward net without the use of GAN.
    gan_path = None

    locals()  # quieten flake8


def uncurry_pad_i2_of_4(arg: Any) -> tuple[None, None, Any, None]:
    """Pads output with None such that input arg is at index 2 in the output 4-tuple.
    arg -> (None, None, arg, None)"""
    tuple = (None, None, arg, None)
    return tuple


@interpret_ex.main
def interpret(
    common: dict,  # Sacred magic: This dict will contain the sacred config for common.
    reward_path: Optional[str],
    rollout_path: str,
    n_expert_demos: Optional[int],
    limit_num_obs: int,
    pyplot: bool,
    vis_scale: int,
    vis_type: str,
    layer_name: str,
    num_features: int,
    gan_path: Optional[str],
):
    """Run interpretability techniques.

    Args:
        For explanation of params see sacred config,
        i.e. comments in defaults function above.
    """
    if limit_num_obs <= 0:
        raise ValueError(
            f"limit_num_obs must be positive, got {limit_num_obs}. "
            f"It used to be possible to specify -1 to use all observations, however "
            f"I don't think we actually ever want to use all so this is currently not "
            f"implemented."
        )
    # Set up imitation-style logging.
    custom_logger, log_dir = common_config.setup_logging()
    wandb_logging = "wandb" in common["log_format_strs"]

    if pyplot:
        matplotlib.use("TkAgg")

    device = "cuda" if th.cuda.is_available() else "cpu"

    # Load reward not pytorch module
    rew_net = th.load(str(reward_path), map_location=th.device(device))

    if gan_path is None:
        # Imitation reward nets have 4 input args, lucent expects models to only have 1.
        # This wrapper makes it so rew_net accepts a single input which is a
        # transition tensor.
        rew_net = TensorTransitionModel(rew_net)
    else:  # Use GAN
        # Combine rew net with GAN.
        raise NotImplementedError()

    rew_net.eval()  # Eval for visualization.

    # This is due to how lucent works
    # if vis_type == "traditional":
    #     rew_net = FourDimOutput(rew_net)
    # Argument venv not necessary, as it is ignored for SupvervisedRewardNet
    # rew_fn = load_reward("SupervisedRewardNet", reward_path, venv=None)

    custom_logger.log("Available layers:")
    custom_logger.log(get_model_layers(rew_net))

    # Load the inputs into the model that are used to do dimensionality reduction and
    # getting the shape of activations.
    if gan_path is None:  # This is when analyzing a reward net only.
        # Load trajectories and transform them into transition tensors.
        transition_tensor_dataloader = rollouts_to_dataloader(
            rollouts_paths=rollout_path,
            num_acts=15,
            batch_size=limit_num_obs,
        )
        # For dim reductions and gettings activations in LayerNMF we want one big batch
        # of limit_num_obs transitions. So, we simply use that as batch_size and sample
        # the first element from the dataloader.
        # for batch in transition_tensor_dataloader:
        #     transition_tensor = batch
        #     break
        inputs = next(iter(transition_tensor_dataloader))
    else:  # When using GAN.
        # Inputs should be some samples of input vectors? Not sure if this is the best
        # way to do this, there might be better options.
        # The important part is that lucent expects 4D tensors as inputs, so increase
        # dimensionality accordingly.
        raise NotImplementedError()

    # The model to analyse should be a torch module that takes a single input.
    # In our case this is one of the following:
    # A reward net that accepts transition tensors
    # A combo of GAN and reward net that accepts latent inputs vectors
    model_to_analyse = rew_net
    nmf = LayerNMF(
        model=model_to_analyse,
        features=num_features,
        layer_name=layer_name,
        # layer_name="cnn_regressor_avg_pool",
        # "obses", i.e. input samples are used for dim reduction (if features is not
        # None) and for determining the shape of the features.
        obses=inputs,
        activation_fn="sigmoid",
    )

    custom_logger.log(f"Dimensionality reduction (to, from): {nmf.channel_dirs.shape}")

    # Visualization
    num_features = nmf.features
    rows, columns = 1, num_features
    if pyplot:
        fig = plt.figure(figsize=(columns * 2, rows * 2))  # width, height in inches

    # Do Visualziation
    if vis_type == "traditional":
        # List of transforms
        transforms = [
            transform.jitter(2),  # Jitters input by 2 pixel
            # Input into model should be 4 tuple, where next_state (3rd arg) is the
            # observation and other inputs are ignored.
            # uncurry_pad_i2_of_4,
        ]

        img = nmf.vis_traditional(transforms=transforms)
        # Set of images, one for each feature, add each to plot
        for feature_i in range(img.shape[0]):
            sub_img = img[feature_i]
            plot_img(
                columns,
                custom_logger,
                feature_i,
                fig,
                sub_img,
                pyplot,
                rows,
                vis_scale,
                wandb_logging,
            )
    elif vis_type == "dataset":
        for feature_i in range(num_features):
            custom_logger.log(f"Feature {feature_i}")

            img, indices = nmf.vis_dataset_thumbnail(
                feature=feature_i, num_mult=4, expand_mult=1
            )

            plot_img(
                columns,
                custom_logger,
                feature_i,
                fig,
                img,
                pyplot,
                rows,
                vis_scale,
                wandb_logging,
            )
    else:
        raise ValueError(f"Unknown vis_type: {vis_type}.")

        # show()
    if pyplot:
        plt.show()
    custom_logger.log("Done with dataset visualization.")


def plot_img(
    columns, custom_logger, feature_i, fig, img, pyplot, rows, vis_scale, wandb_logging
):
    """Plot the passed image to pyplot and wandb as appropriate."""
    _wandb_log(custom_logger, feature_i, img, vis_scale, wandb_logging)
    if pyplot:
        fig.add_subplot(rows, columns, feature_i + 1)
        plt.imshow(img)


def _wandb_log(
    custom_logger, feature_i: int, img: np.ndarray, vis_scale: int, wandb_logging: bool
):
    """Plot to wandb if wandb logging is enabled."""
    if wandb_logging:
        p_img = Image.fromarray(np.uint8(img * 255), mode="RGB").resize(
            size=(img.shape[0] * vis_scale, img.shape[1] * vis_scale),
            resample=Image.NEAREST,
        )
        wb_img = wandb.Image(p_img, caption=f"Feature {feature_i}")
        custom_logger.record(f"feature_{feature_i}", wb_img)
        # Can't re-use steps unfortunately, so each feature img gets its own step.
        custom_logger.dump(step=feature_i)


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

import os.path as osp
from typing import Optional

from PIL import Image
from imitation.scripts.common import common as common_config
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import transform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred.observers import FileStorageObserver
import torch as th
import wandb

from reward_preprocessing.common.utils import (
    TensorTransitionWrapper,
    rollouts_to_dataloader,
    tensor_to_transition,
)
from reward_preprocessing.scripts.config.interpret import interpret_ex
from reward_preprocessing.vis.reward_vis import LayerNMF


@interpret_ex.main
def interpret(
    common: dict,
    reward_path: str,
    # TODO: I think at some point it would be cool if this was optional, since these
    # are only used for dataset visualization, dimensionality reduction, and
    # determining the shape of the features. In the case that we aren't doing the first
    # two, we could determine the shape of the features some other way.
    # This would especially help when incorporating a GAN into the procedure, because
    # here the notion of a "rollout" as input into the whole pipeline doesn't make as
    # much sense.
    rollout_path: str,
    limit_num_obs: int,
    pyplot: bool,
    vis_scale: int,
    vis_type: str,
    layer_name: str,
    num_features: Optional[int],
    gan_path: Optional[str],
):
    """Run visualization for interpretability.

    Args:
        common:
            Sacred magic: This dict will contain the sacred config settings for the
            sub_section 'common' in the sacred config. These settings are defined in the
            sacred ingredient 'common' in imitation.scripts.common.
        reward_path: Path to the learned supervised reward net.
        rollout_path:
            Rollouts to use for dataset visualization, dimensionality
            reduction, and determining the shape of the features.
        limit_num_obs:
            Limit how many of the transitions from `rollout_path` are used for
            dimensionality reduction. The RL Vision paper uses "a few thousand"
            sampled infrequently from rollouts.
        pyplot: Whether to plot images as pyplot figures.
        vis_scale: Scale the plotted images by this factor.
        vis_type:
            Type of visualization to use. Either "traditional" for gradient-based
            visualization of activations, or "dataset" for dataset visualization.
        layer_name:
            Name of the layer to visualize. To figure this out run this script and the
            available layers in the loaded model will be printed. Available layers will
            be those that are "named" layers in the torch Module, i.e. those that are
            declared as attributes in the torch Module.
        num_features:
            Number of features to use for visualization. The activations will be reduced
            to this size using NMF. If None, performs no dimensionality reduction.
        gan_path:
            Path to the GAN model. This is used to regularize the output of the
            visualization. If None simply visualize reward net without the use
            of a GAN in the pipeline.
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
        rew_net = TensorTransitionWrapper(rew_net)
    else:  # Use GAN
        # Combine rew net with GAN.
        raise NotImplementedError()

    rew_net.eval()  # Eval for visualization.

    custom_logger.log("Available layers:")
    custom_logger.log(get_model_layers(rew_net))

    # Load the inputs into the model that are used to do dimensionality reduction and
    # getting the shape of activations.
    if gan_path is None:  # This is when analyzing a reward net only.
        # Load trajectories and transform them into transition tensors.
        # TODO: Seeding so the randomly shuffled subset is always the same.
        transition_tensor_dataloader = rollouts_to_dataloader(
            rollouts_paths=rollout_path,
            num_acts=15,
            batch_size=limit_num_obs,
            # This is an upper bound of the number of trajectories we need, since every
            # trajectory has at least 1 transition.
            n_trajectories=limit_num_obs,
        )
        # For dim reductions and gettings activations in LayerNMF we want one big batch
        # of limit_num_obs transitions. So, we simply use that as batch_size and sample
        # the first element from the dataloader.
        inputs: th.Tensor = next(iter(transition_tensor_dataloader))
        inputs = inputs.to(device)
        # Ensure loaded data is FloatTensor and not DoubleTensor.
        inputs = inputs.float()
    else:  # When using GAN.
        # Inputs should be some samples of input vectors? Not sure if this is the best
        # way to do this, there might be better options.
        # The important part is that lucent expects 4D tensors as inputs, so increase
        # dimensionality accordingly.
        raise NotImplementedError()

    # The model to analyse should be a torch module that takes a single input, which
    # should be a torch Tensor.
    # In our case this is one of the following:
    # - A reward net that has been wrapped, so it accepts transition tensors.
    # - A combo of GAN and reward net that accepts latent inputs vectors. (TODO)
    model_to_analyse = rew_net
    nmf = LayerNMF(
        model=model_to_analyse,
        features=num_features,
        layer_name=layer_name,
        # input samples are used for dim reduction (if features is not
        # None) and for determining the shape of the features.
        model_inputs_preprocess=inputs,
        activation_fn="sigmoid",
    )

    custom_logger.log(f"Dimensionality reduction (to, from): {nmf.channel_dirs.shape}")
    # If these are equal, then of course there is no actual reduction.

    num_features = nmf.channel_dirs.shape[0]
    rows, columns = 1, num_features
    if pyplot:
        fig = plt.figure(figsize=(columns * 2, rows * 2))  # width, height in inches
    else:
        fig = None

    # Visualize
    if vis_type == "traditional":
        # List of transforms
        transforms = [
            transform.jitter(2),  # Jitters input by 2 pixel
        ]

        opt_transitions = nmf.vis_traditional(transforms=transforms)
        # This gives as an array that optimizes the objectives, in the shape of the
        # input which is a transition tensor. However, lucent helpfully transposes the
        # output such that the channel dimension is last. Our functions expect channel
        # dim before spatial dims, so we need to transpose it back.
        opt_transitions = opt_transitions.transpose(0, 3, 1, 2)
        # Split the optimized transitions, one for each feature, into separate
        # observations and actions. This function only works with torch tensors.
        obs, acts, next_obs = tensor_to_transition(th.tensor(opt_transitions))
        # obs and next_obs output have channel dim last.
        # acts is output as one-hot vector.

        # Set of images, one for each feature, add each to plot
        for feature_i in range(next_obs.shape[0]):
            sub_img = next_obs[feature_i]
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

    if pyplot:
        plt.show()
    custom_logger.log("Done with dataset visualization.")


def plot_img(
    columns,
    custom_logger,
    feature_i,
    fig: Optional,
    img,
    pyplot,
    rows,
    vis_scale,
    wandb_logging,
):
    """Plot the passed image to pyplot and wandb as appropriate."""
    _wandb_log(custom_logger, feature_i, img, vis_scale, wandb_logging)
    if fig is not None and pyplot:
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

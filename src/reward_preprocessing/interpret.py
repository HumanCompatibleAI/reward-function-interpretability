import os.path as osp
from typing import Optional, Tuple, Union

from PIL import Image
from imitation.scripts.common import common as common_config
from imitation.util.logger import HierarchicalLogger
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import transform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred.observers import FileStorageObserver
import torch as th
import wandb

from reward_preprocessing.common.utils import (
    RewardGeneratorCombo,
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
    gan_path: Optional[str] = None,
    l2_coeff: Optional[float] = None,
    img_save_path: Optional[str] = None,
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
        l2_coeff:
            Strength with which to penalize the L2 norm of generated latent vector
            "visualizations" of a GAN-reward model combination. If gan_path is not None,
            this must also not be None.
        img_save_path:
            Directory to save images in. Must end in a /. If None, do not save images.
    """
    if limit_num_obs <= 0:
        raise ValueError(
            f"limit_num_obs must be positive, got {limit_num_obs}. "
            f"It used to be possible to specify -1 to use all observations, however "
            f"I don't think we actually ever want to use all so this is currently not "
            f"implemented."
        )
    if vis_type not in ["dataset", "traditional"]:
        raise ValueError(f"Unknown vis_type: {vis_type}")
    if vis_type == "dataset" and gan_path is not None:
        raise ValueError("GANs cannot be used with dataset visualization.")
    if gan_path is not None and l2_coeff is None:
        raise ValueError("When GANs are used, l2_coeff must be set.")
    if img_save_path is not None and img_save_path[-1] != "/":
        raise ValueError("img_save_path is not a directory, does not end in /")

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
        gan = th.load(gan_path, map_location=th.device(device))
        rew_net = RewardGeneratorCombo(reward_net=rew_net, generator=gan.generator)

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
        # Inputs are GAN samples
        samples = gan.sample(limit_num_obs)
        inputs = samples[:, :, None, None]
        inputs = inputs.to(device)
        inputs = inputs.float()

    # The model to analyse should be a torch module that takes a single input, which
    # should be a torch Tensor.
    # In our case this is one of the following:
    # - A reward net that has been wrapped, so it accepts transition tensors.
    # - A combo of GAN and reward net that accepts latent inputs vectors.
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
        col_mult = 4 if vis_type == "traditional" else 2
        # figsize is width, height in inches
        fig = plt.figure(figsize=(columns * col_mult, rows * 2))
    else:
        fig = None

    # Visualize
    if vis_type == "traditional":

        if gan_path is None:
            # List of transforms
            transforms = [
                transform.jitter(2),  # Jitters input by 2 pixel
            ]

            opt_transitions = nmf.vis_traditional(transforms=transforms)
            # This gives as an array that optimizes the objectives, in the shape of the
            # input which is a transition tensor. However, lucent helpfully transposes
            # the output such that the channel dimension is last. Our functions expect
            # channel dim before spatial dims, so we need to transpose it back.
            opt_transitions = opt_transitions.transpose(0, 3, 1, 2)
            # Split the optimized transitions, one for each feature, into separate
            # observations and actions. This function only works with torch tensors.
            obs, acts, next_obs = tensor_to_transition(th.tensor(opt_transitions))
            # obs and next_obs output have channel dim last.
            # acts is output as one-hot vector.

        else:
            # We do not require the latent vectors to be transformed before optimizing.
            # However, we do regularize the L2 norm of latent vectors, to ensure the
            # resulting generated images are realistic.
            opt_latent = nmf.vis_traditional(
                transforms=[],
                l2_coeff=l2_coeff,
                l2_layer_name="input",
            )
            # Now, we put the latent vector thru the generator to produce transition
            # tensors that we can get observations, actions, etc out of
            opt_latent = np.mean(opt_latent, axis=(1, 2))
            opt_latent_th = th.from_numpy(opt_latent).to(th.device(device))
            opt_transitions = gan.generator(opt_latent_th)
            obs, acts, next_obs = tensor_to_transition(opt_transitions)

        # Set of images, one for each feature, add each to plot
        for feature_i in range(next_obs.shape[0]):
            sub_img_obs = obs[feature_i].detach().cpu().numpy()
            sub_img_next_obs = next_obs[feature_i].detach().cpu().numpy()
            plot_img(
                columns,
                custom_logger,
                feature_i,
                fig,
                (sub_img_obs, sub_img_next_obs),
                pyplot,
                rows,
                vis_scale,
                wandb_logging,
            )
            if img_save_path is not None:
                obs_PIL = array_to_image(sub_img_obs, vis_scale)
                obs_PIL.save(img_save_path + f"{feature_i}_obs.png")
                next_obs_PIL = array_to_image(sub_img_next_obs, vis_scale)
                next_obs_PIL.save(img_save_path + f"{feature_i}_next_obs.png")
                custom_logger.log(
                    f"Saved feature {feature_i} viz in dir {img_save_path}."
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

    if pyplot:
        plt.show()
    custom_logger.log("Done with dataset visualization.")


def array_to_image(arr: np.ndarray, scale: int) -> Image:
    """Take numpy array on [0,1] scale, return PIL image."""
    return Image.fromarray(np.uint8(arr * 255), mode="RGB").resize(
        size=(arr.shape[0] * scale, arr.shape[1] * scale),
        resample=Image.NEAREST,
    )


def plot_img(
    columns: int,
    custom_logger: HierarchicalLogger,
    feature_i: int,
    fig: Optional[matplotlib.figure.Figure],
    img: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    pyplot: bool,
    rows: int,
    vis_scale: int,
    wandb_logging: bool,
):
    """Plot the passed image(s) to pyplot and wandb as appropriate."""
    _wandb_log(custom_logger, feature_i, img, vis_scale, wandb_logging)
    if pyplot:
        if isinstance(img, tuple):
            img_obs = img[0]
            img_next_obs = img[1]
            fig.add_subplot(rows, columns, 2 * feature_i + 1)
            plt.imshow(img_obs)
            fig.add_subplot(rows, columns, 2 * feature_i + 2)
            plt.imshow(img_next_obs)
        else:
            fig.add_subplot(rows, columns, feature_i + 1)
            plt.imshow(img)


def _wandb_log(
    custom_logger: HierarchicalLogger,
    feature_i: int,
    img: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    vis_scale: int,
    wandb_logging: bool,
):
    """Plot to wandb if wandb logging is enabled."""
    if wandb_logging:
        if isinstance(img, tuple):
            img_obs = img[0]
            img_next_obs = img[1]
            # TODO(df): check if I have to dump between these
            _wandb_log_(img_obs, vis_scale, feature_i, "obs", custom_logger)
            _wandb_log_(img_next_obs, vis_scale, feature_i, "next_obs", custom_logger)
        else:
            _wandb_log_(img, vis_scale, feature_i, "dataset_vis", custom_logger)

        # Can't re-use steps unfortunately, so each feature img gets its own step.
        custom_logger.dump(step=feature_i)


def _wandb_log_(
    arr: np.ndarray,
    scale: int,
    feature: int,
    img_type: str,
    logger: HierarchicalLogger,
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


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

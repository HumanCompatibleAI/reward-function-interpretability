import io
import os.path as osp
from typing import Optional, Tuple, Union

import PIL.Image
from imitation.scripts.common import common as common_config
from imitation.util.logger import HierarchicalLogger
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import transform
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common.utils import (
    RewardGeneratorCombo,
    TensorTransitionWrapper,
    array_to_image,
    log_img_wandb,
    rollouts_to_dataloader,
    tensor_to_transition,
)
from reward_preprocessing.scripts.config.interpret import interpret_ex
from reward_preprocessing.vis.reward_vis import LayerNMF


def get_action_meaning(action_id: int):
    """Get a human-understandable name for an action. Currently only supports coinrun."""
    # Taken from get_combos() in coinrun.env.BaseProcgenEnv
    mapping = [
        ("LEFT", "DOWN"),
        ("LEFT",),
        ("LEFT", "UP"),
        ("DOWN",),
        ("NOOP",),
        ("UP",),
        ("RIGHT", "DOWN"),
        ("RIGHT",),
        ("RIGHT", "UP"),
        ("D",),
        ("A",),
        ("W",),
        ("S",),
        ("Q",),
        ("E",),
    ]
    action = mapping[action_id]
    return ", ".join(action)


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
        matplotlib.use("Agg")

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
    rows, columns = 2, num_features
    if pyplot:
        col_mult = 2 if vis_type == "traditional" else 1
        # figsize is width, height in inches
        fig = plt.figure(figsize=(int(columns * col_mult), int(rows * 2)))
    else:
        fig = None

    # Visualize
    if vis_type == "traditional":

        if gan_path is None:
            # List of transforms
            transforms = [
                transform.jitter(16),
            ]

            # This does the actual interpretability, i.e. it calculates the
            # visualizations.
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
                l2_layer_name="generator_network_latent_vec",
            )
            # Now, we put the latent vector thru the generator to produce transition
            # tensors that we can get observations, actions, etc out of
            opt_latent = np.mean(opt_latent, axis=(1, 2))
            opt_latent_th = th.from_numpy(opt_latent).to(th.device(device))
            opt_transitions = gan.generator(opt_latent_th)
            obs, acts, next_obs = tensor_to_transition(opt_transitions)

        # Use numpy from here.
        obs = obs.detach().cpu().numpy()
        next_obs = next_obs.detach().cpu().numpy()

        # Set of images, one for each feature, add each to plot
        for feature_i in range(next_obs.shape[0]):
            sub_img_obs = obs[feature_i]
            sub_img_next_obs = next_obs[feature_i]
            _log_single_transition_wandb(
                custom_logger,
                feature_i,
                (sub_img_obs, sub_img_next_obs),
                vis_scale,
                wandb_logging,
            )
            _plot_img(
                columns,
                feature_i,
                num_features,
                fig,
                (sub_img_obs, sub_img_next_obs),
                pyplot,
                rows,
            )
            if img_save_path is not None:
                obs_PIL = array_to_image(sub_img_obs, vis_scale)
                obs_PIL.save(img_save_path + f"{feature_i}_obs.png")
                next_obs_PIL = array_to_image(sub_img_next_obs, vis_scale)
                next_obs_PIL.save(img_save_path + f"{feature_i}_next_obs.png")
                custom_logger.log(
                    f"Saved feature {feature_i} viz in dir {img_save_path}."
                )
        # This greatly improves the spacing of subplots for the feature overview plot.
        plt.tight_layout()
        # Take the matplotlib plot containing all visualizations and log it as a single
        # image in wandb.
        # We do this, so we have both the individual feature visualizations (logged
        # above) in case we need them and the overview plot, which is a bit more useful.
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        full_plot_img = PIL.Image.open(img_buf)
        log_img_wandb(
            img=full_plot_img,
            caption="Feature Overview",
            wandb_key="feature_overview",
            scale=vis_scale,
            logger=custom_logger,
        )
        custom_logger.dump(step=num_features)

    elif vis_type == "dataset":
        for feature_i in range(num_features):
            custom_logger.log(f"Feature {feature_i}")

            img, indices = nmf.vis_dataset_thumbnail(
                feature=feature_i, num_mult=4, expand_mult=1
            )

            _log_single_transition_wandb(
                custom_logger, feature_i, img, vis_scale, wandb_logging
            )
            _plot_img(
                columns,
                feature_i,
                num_features,
                fig,
                img,
                pyplot,
                rows,
            )

    if pyplot:
        plt.show()
    custom_logger.log("Done with visualization.")


def _plot_img(
    columns: int,
    feature_i: int,
    num_features: int,
    fig: Optional[matplotlib.figure.Figure],
    img: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    pyplot: bool,
    rows: int,
):
    """Plot the passed image(s) with pyplot, if pyplot is enabled."""
    if fig is not None and pyplot:
        if isinstance(img, tuple):
            img_obs = img[0]
            img_next_obs = img[1]
            obs_i = feature_i + 1
            f = fig.add_subplot(rows, columns, obs_i)
            # This title will be at every column
            f.set_title(f"Feature {feature_i}")
            if obs_i == 1:  # First image
                f.set_ylabel("obs")
            plt.imshow(img_obs)
            # In 2-column layout, next_obs should be logged below the obs.
            next_obs_i = obs_i + num_features
            f = fig.add_subplot(rows, columns, next_obs_i)
            if obs_i == 1:  # f is first image of the second row
                f.set_ylabel("next_obs")
            plt.imshow(img_next_obs)
        else:
            fig.add_subplot(rows, columns, feature_i + 1)
            plt.imshow(img)


def _log_single_transition_wandb(
    custom_logger: HierarchicalLogger,
    feature_i: int,
    img: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    vis_scale: int,
    wandb_logging: bool,
):
    """Plot visualizations to wandb if wandb logging is enabled. Images will be logged
    as separate media in wandb, one for each feature."""
    if wandb_logging:
        if isinstance(img, tuple):
            img_obs = img[0]
            img_next_obs = img[1]

            log_img_wandb(
                img=img_obs,
                caption=f"Feature {feature_i}, obs",
                wandb_key=f"feature_{feature_i}_obs",
                scale=vis_scale,
                logger=custom_logger,
            )
            log_img_wandb(
                img=img_next_obs,
                caption=f"Feature {feature_i}, next_obs",
                wandb_key=f"feature_{feature_i}_next_obs",
                scale=vis_scale,
                logger=custom_logger,
            )
        else:
            log_img_wandb(
                img=img,
                caption=f"Feature {feature_i}",
                wandb_key=f"dataset_vis_{feature_i}",
                scale=vis_scale,
                logger=custom_logger,
            )

        # Can't re-use steps unfortunately, so each feature img gets its own step.
        custom_logger.dump(step=feature_i)


def main():
    observer = FileStorageObserver(osp.join("output", "sacred", "interpret"))
    interpret_ex.observers.append(observer)
    interpret_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main()

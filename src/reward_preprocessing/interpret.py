import io
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
from imitation.rewards.reward_nets import RewardNet
from imitation.scripts.common import common as common_config
from imitation.util.logger import HierarchicalLogger
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import transform
from lucent.optvis.param.spatial import pixel_image
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
    ndarray_to_transition,
    rollouts_to_dataloader,
    tensor_to_transition,
)
from reward_preprocessing.scripts.config.interpret import interpret_ex
from reward_preprocessing.vis.reward_vis import LayerNMF


def _get_action_meaning(action_id: int):
    """Get a human-understandable name for an action.
    Currently, only supports coinrun.
    """
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


def _determine_features_are_actions(nmf: LayerNMF, layer_name: str) -> bool:
    """This function decides whether the features that we are visualizing correspond to
    the different actions.
    In interpret we either (1) visualize features directly, (2) perform dimensionality
    reduction on the features and visualize these reduced features. In the case that we
    are visualizing a reward net which outputs a separate reward for each action in the
    last layer, (1) corresponds to having one visualization per action. This function
    can be used to decided whether we are in this special case in order to e.g. log the
    human-understandable action name instead of the feature index."""
    # This is the heuristic for determining whether features are actions:
    # - If there is no dim reduction
    # - If the name of the layer suggests that we are analysing the final layer of a
    #   reward net
    # - If the number of features is 15 since that is the number of actions in all
    #   procgen games
    return (
        nmf.channel_dirs.shape[0] == nmf.channel_dirs.shape[1]
        and layer_name.endswith("dense_final")
        and nmf.channel_dirs.shape[0] == 15
    )


# TODO: break up the body of this into separate functions
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
    l2_coeff: Optional[float],
    img_save_path: Optional[str],
    reg: Dict[str, Dict[str, Any]],
    num_random_samples: Optional[int],
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
        pyplot:
            Whether to plot visualizations as pyplot figures. Set to False when running
            interpret in a non-GUI environment, such as the cluster. In that case, use
            wandb logging or save images to disk.
        vis_scale: Scale the plotted images by this factor.
        vis_type:
            Type of visualization to use. Either "traditional" for gradient-based
            visualization of activations, "dataset" for dataset visualization, or
            "dataset_traditional" to start with dataset visualization, then do
            gradient-based visualization using the dataset visualizations as
            initializations.
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
        reg:
            Regularization settings. See reward_preprocessing.scripts.config.interpret
            for defaults.
        num_random_samples:
            Number of random pixel observations to feed to the reward net, to test if
            the network gives them sensible rewards.
    """
    if limit_num_obs <= 0:
        raise ValueError(
            f"limit_num_obs must be positive, got {limit_num_obs}. "
            f"It used to be possible to specify -1 to use all observations, however "
            f"I don't think we actually ever want to use all so this is currently not "
            f"implemented."
        )
    if vis_type not in ["dataset", "traditional", "dataset_traditional"]:
        raise ValueError(f"Unknown vis_type: {vis_type}")
    if vis_type == "dataset" and gan_path is not None:
        raise ValueError(f"GANs cannot be used with {vis_type} visualization.")
    if gan_path is not None and l2_coeff is None:
        raise ValueError("When GANs are used, l2_coeff must be set.")
    if img_save_path is not None and img_save_path[-1] != "/":
        raise ValueError("img_save_path is not a directory, does not end in /")

    # Set up imitation-style logging.
    custom_logger, log_dir = common_config.setup_logging()
    wandb_logging = "wandb" in common["log_format_strs"]

    if pyplot:
        # "TkAgg" is a GUI backend, doesn't work on the cluster.
        matplotlib.use("TkAgg")

    device = "cuda" if th.cuda.is_available() else "cpu"

    # Load reward net pytorch module
    rew_net = th.load(str(reward_path), map_location=th.device(device))

    if gan_path is None:
        # Imitation reward nets have 4 input args, lucent expects models to only have 1.
        # This wrapper makes it so rew_net accepts a single input which is a
        # transition tensor.
        model_to_analyse = TensorTransitionWrapper(rew_net)
    else:  # Use GAN
        # Combine rew net with GAN.
        gan = th.load(gan_path, map_location=th.device(device))
        model_to_analyse = RewardGeneratorCombo(
            rew_net=rew_net, generator=gan.generator
        )

    model_to_analyse.eval()  # Eval for visualization.

    custom_logger.log("Available layers:")
    custom_logger.log(get_model_layers(model_to_analyse))

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
    nmf = LayerNMF(
        model=model_to_analyse,
        features=num_features,
        layer_name=layer_name,
        # input samples are used for dim reduction (if features is not
        # None) and for determining the shape of the features.
        model_inputs_preprocess=inputs,
        activation_fn="relu",
    )

    # If these are equal, then of course there is no actual reduction.
    custom_logger.log(f"Dimensionality reduction (to, from): {nmf.channel_dirs.shape}")

    num_features = nmf.channel_dirs.shape[0]
    rows, columns = 2, num_features
    col_mult = 2 if vis_type == "traditional" else 1
    # figsize is width, height in inches
    fig = plt.figure(figsize=(int(columns * col_mult), int(rows * 2)))

    # Visualize
    if vis_type == "traditional":
        if gan_path is None:
            # List of transforms
            transforms = _determine_transforms(reg)
            # This does the actual interpretability, i.e. it calculates the
            # visualizations.
            opt_transitions = nmf.vis_traditional(transforms=transforms)
            # This gives us an array that optimizes the objectives, in the shape of
            # the input which is a transition tensor. However, lucent helpfully
            # transposes the output such that the channel dimension is last. Our
            # functions expect channel dim before spatial dims, so we need to
            # transpose it back.
            opt_transitions = opt_transitions.transpose(0, 3, 1, 2)
            # In the following we need opt_transitions to be a pytorch tensor.
            opt_transitions = th.tensor(opt_transitions)
            # Split the optimized transitions, one for each feature, into separate
            # observations and actions. This function only works with torch tensors.
            obs, acts, next_obs = tensor_to_transition(opt_transitions)
            # obs and next_obs output have channel dim last.
            # acts is output as one-hot vector.
        else:
            # We do not require the latent vectors to be transformed before optimizing.
            # However, we do regularize the L2 norm of latent vectors, to ensure the
            # resulting generated images are realistic.
            z_dim_as_expected = (
                isinstance(gan.z_dim, tuple)
                and len(gan.z_dim) == 1
                and isinstance(gan.z_dim[0], int)
            )
            if not z_dim_as_expected:
                error_string = (
                    "interpret.py expects the GAN's latent input shape to "
                    + f"be a tuple of length 1, instead it is {gan.z_dim}."
                )
                raise TypeError(error_string)
            # ensure visualization doesn't treat the latent vector as an image.
            latent_shape = (num_features, gan.z_dim[0], 1, 1)

            def param_f():
                return pixel_image(shape=latent_shape)

            opt_latent = nmf.vis_traditional(
                transforms=[],
                l2_coeff=l2_coeff,
                l2_layer_name="generator_network_latent_vec",
                param_f=param_f,
            )
            # Now, we put the latent vector thru the generator to produce transition
            # tensors that we can get observations, actions, etc out of
            opt_latent = np.squeeze(opt_latent, axis=(1, 2))
            # ^ squeeze out extraneous "height" and "width" dimensions
            opt_latent_th = th.from_numpy(opt_latent).to(th.device(device))
            opt_transitions = gan.generator(opt_latent_th)
            obs, acts, next_obs = tensor_to_transition(opt_transitions)

        plot_trad_vis(
            obs,
            next_obs,
            device,
            rew_net,
            layer_name,
            num_features,
            nmf,
            fig,
            rows,
            columns,
            vis_scale,
            img_save_path,
            wandb_logging,
            custom_logger,
        )

    elif vis_type == "dataset":
        for feature_i in range(num_features):
            custom_logger.log(f"Feature {feature_i}")

            dataset_thumbnails, indices = nmf.vis_dataset_thumbnail(
                feature=feature_i,
                num_mult=4,
                expand_mult=1,
            )

            if nmf.reducer is None:
                # print out rewards
                flat_indices = []
                for index_list in indices:
                    flat_indices += index_list
                obses, _, next_obses = tensor_to_transition(inputs[flat_indices])
                feature_i_rep = th.Tensor([feature_i] * len(flat_indices)).long()
                action_i_tens = th.nn.functional.one_hot(
                    feature_i_rep, num_classes=num_features
                ).to(device)
                rewards = rew_net(obses, action_i_tens, next_obses, done=None)
                custom_logger.log(f"Rewards for feature {feature_i}: {rewards}")

            # remove opacity channel from dataset thumbnails
            np_trans_tens = dataset_thumbnails[:-1, :, :]

            obs, _, next_obs = ndarray_to_transition(np_trans_tens)

            _log_single_transition_wandb(
                custom_logger, feature_i, (obs, next_obs), vis_scale, wandb_logging
            )
            _plot_img(
                columns,
                feature_i,
                num_features,
                fig,
                (obs, next_obs),
                rows,
            )

            if img_save_path is not None:
                obs_PIL = array_to_image(obs, vis_scale)
                obs_PIL.save(img_save_path + f"{feature_i}_obs.png")
                next_obs_PIL = array_to_image(next_obs, vis_scale)
                next_obs_PIL.save(img_save_path + f"{feature_i}_next_obs.png")
                custom_logger.log(
                    f"Saved feature {feature_i} viz in dir {img_save_path}."
                )

    elif vis_type == "dataset_traditional":
        best_transitions = []
        for feature_i in range(num_features):
            custom_logger.log(f"Feature {feature_i}")
            _, indices = nmf.vis_dataset_thumbnail(
                feature=feature_i,
                num_mult=2,
                expand_mult=1,
            )

            assert nmf.reducer is None
            best_transitions.append(th.Tensor(inputs[indices[0][0]][None, :, :, :]))

        dataset_vis = th.cat(best_transitions)
        dataset_vis_clone = th.clone(dataset_vis).detach()

        def pixel_image_start_best():
            tensor = dataset_vis.to(device).requires_grad_(True)
            return [tensor], lambda: tensor

        transforms = _determine_transforms(reg)
        opt_dataset = nmf.vis_traditional(
            transforms=transforms,
            param_f=pixel_image_start_best,
            l2_diff_coeff=3e-1,
            l2_diff_tensor=dataset_vis_clone,
        )

        opt_dataset = opt_dataset.transpose(0, 3, 1, 2)
        opt_dataset = th.tensor(opt_dataset).to(device)
        obs, acts, next_obs = tensor_to_transition(opt_dataset)

        plot_trad_vis(
            obs,
            next_obs,
            device,
            rew_net,
            layer_name,
            num_features,
            nmf,
            fig,
            rows,
            columns,
            vis_scale,
            img_save_path,
            wandb_logging,
            custom_logger,
        )

    if num_random_samples is not None:
        random_rewards(
            num_random_samples, rew_net, obs, num_features, device, custom_logger
        )

    if pyplot:
        plt.show()
    custom_logger.log("Done with visualization.")


def plot_trad_vis(
    obs: th.Tensor,
    next_obs: th.Tensor,
    device: Union[str, th.device],
    rew_net: RewardNet,
    layer_name: str,
    num_features: int,
    nmf,
    fig,
    rows,
    columns,
    vis_scale,
    img_save_path,
    wandb_logging,
    custom_logger,
):
    """Print out rewards and plot results of gradient-based visualization.

    Key difference from dataset vis plotting seems to be the use of sub-images?
    To be honest, I (Daniel Filan) wrote this docstring in a state of not really
    understanding why this code is the way it is.

    Hopefully the limited degree of type information available is still useful.
    """
    # What reward does the model output for these generated transitions?
    # (done isn't used in the reward function)
    # There are three possible options here:
    # - The reward net does not use action -> it does not matter what we pass as
    #   action.
    # - The reward net does use action, and we are optimizing an intermediate layer
    #   -> since action is only used on the final layer (to choose which of the 15
    #   heads has the correct reward), it does not matter what we pass as action.
    # - The reward net does use action, and we are optimizing the final layer
    #  -> the action index of the action corresponds to the index of the feature.
    # Note that since actions is only used to choose which head to use, there are no
    # gradients from the reward to the action. Consequently, acts in opt_latent is
    # meaningless.

    action_nums = th.tensor(list(range(num_features))).to(device)
    actions = th.nn.functional.one_hot(action_nums, num_classes=num_features)

    assert len(actions) == len(obs)
    rews = rew_net(obs.to(device), actions, next_obs.to(device), done=None)
    custom_logger.log(f"Rewards: {rews}")

    # Use numpy from here.
    obs = obs.detach().cpu().numpy()
    next_obs = next_obs.detach().cpu().numpy()
    rews = rews.detach().cpu().numpy()

    # We want to plot the name of the action, if applicable.
    features_are_actions = _determine_features_are_actions(nmf, layer_name)

    # Set of images, one for each feature, add each to plot
    for feature_i in range(next_obs.shape[0]):
        # Log the rewards
        rew_key = f"rew_feat_{feature_i:02}"
        if features_are_actions:
            rew_key += f"_{_get_action_meaning(action_id=feature_i)}"
        custom_logger.record(rew_key, rews[feature_i])
        # Log the images
        sub_img_obs = obs[feature_i]
        sub_img_next_obs = next_obs[feature_i]
        _log_single_transition_wandb(
            custom_logger,
            feature_i,
            (sub_img_obs, sub_img_next_obs),
            vis_scale,
            wandb_logging,
            features_are_actions,
        )
        _plot_img(
            columns,
            feature_i,
            num_features,
            fig,
            (sub_img_obs, sub_img_next_obs),
            rows,
            features_are_actions,
        )
        if img_save_path is not None:
            obs_PIL = array_to_image(sub_img_obs, vis_scale)
            obs_PIL.save(img_save_path + f"{feature_i}_obs.png")
            next_obs_PIL = array_to_image(sub_img_next_obs, vis_scale)
            next_obs_PIL.save(img_save_path + f"{feature_i}_next_obs.png")
            custom_logger.log(f"Saved feature {feature_i} viz in dir {img_save_path}.")
    # This greatly improves the spacing of subplots for the feature overview plot.
    plt.tight_layout()

    if wandb_logging:
        # Take the matplotlib plot containing all visualizations and log it as a
        # single image in wandb.
        # We do this, so we have both the individual feature visualizations (logged
        # above) in case we need them and the overview plot, which is a bit more
        # useful.
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


def random_rewards(num_samples, rew_net, obs, num_features, device, custom_logger):
    """Prints rewards of random observations."""
    action_nums = th.tensor(list(range(num_features))).to(device)
    actions = th.nn.functional.one_hot(action_nums, num_classes=num_features)

    for _ in range(num_samples):
        rand_obs = th.randn(obs.shape) + 0.5
        rand_next_obs = th.randn(obs.shape) + 0.5
        rand_rews = rew_net(
            rand_obs.to(device).float(),
            actions,
            rand_next_obs.to(device).float(),
            done=None,
        )
        custom_logger.log(f"Rewards of random obs and next obs: {rand_rews}")
        half_rand_rews = rew_net(
            rand_obs.to(device).float(),
            actions,
            rand_obs.to(device).float(),
            done=None,
        )
        custom_logger.log(f"Rewards of random obs and same next obs: {half_rand_rews}")


def _determine_transforms(reg: Dict[str, Dict[str, Any]]) -> List[Callable]:
    """Determine the transforms to use for traditional visualization. Currently, only
    applicable to vis without GAN."""
    return [
        transform.jitter(reg["no_gan"]["jitter"]),
    ]


def _plot_img(
    columns: int,
    feature_i: int,
    num_features: int,
    fig: matplotlib.figure.Figure,
    img: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    rows: int,
    features_are_actions: bool = False,
):
    """Plot the passed image(s) with pyplot, if pyplot is enabled."""
    if isinstance(img, tuple):
        img_obs = img[0]
        img_next_obs = img[1]
        obs_i = feature_i + 1
        f = fig.add_subplot(rows, columns, obs_i)
        title = f"Feature {feature_i}"
        if features_are_actions:
            title += f"\n({_get_action_meaning(feature_i)})"
        # This title will be at every column
        f.set_title(title)
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
    features_are_actions: bool = False,
):
    """Plot visualizations to wandb if wandb logging is enabled. Images will be logged
    as separate media in wandb, one for each feature."""
    if wandb_logging:
        if isinstance(img, tuple):
            img_obs = img[0]
            img_next_obs = img[1]
            caption = f"Feature {feature_i}"
            if features_are_actions:
                caption += f"\n({_get_action_meaning(feature_i)})"
            log_img_wandb(
                img=img_obs,
                caption=f"{caption}\nobs",
                wandb_key=f"feature_{feature_i}_obs",
                scale=vis_scale,
                logger=custom_logger,
            )
            log_img_wandb(
                img=img_next_obs,
                caption=f"{caption}\nnext_obs",
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

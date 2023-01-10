"""Port of lucid.scratch.rl_util to PyTorch. APL2.0 licensed."""
from functools import reduce
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from lucent.optvis.objectives import handle_batch, wrap_objective
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
import numpy as np
import scipy.ndimage as nd
import torch as th

from reward_preprocessing.ext.channel_reducer import ChannelReducer
from reward_preprocessing.vis.attribution import get_activations, get_attr
import reward_preprocessing.vis.objectives as objectives_rfi


def argmax_nd(
    x: np.ndarray,
    axes: List[int],
    *,
    max_rep: Union[int, float] = np.inf,
    max_rep_strict: Optional[bool] = None,
):
    """Return the indices of the maximum value along the given axes.

    Args:
        x: The array to find the maximum of. Size is (N, M, ...).
        axes: The axes to find the maximum along.
        max_rep: The maximum number of times a value can be repeated. If the
            maximum value is repeated more than this, then the returned indices
            will be None.
        max_rep_strict: If True, then the maximum number of repetitions is
            enforced strictly.
    Returns:
        Tuple of numpy arrays, each of size (N,). The values in each array represent the
        x and y coordinates in the original array M, ... dimensions. The coordinates are
        the maximum value for that sample.
        E.g. for x of size (N, M, K), the returned tuple will be (N,) and (N,), where
        the first element of the tuple has values from 0 to M-1, and the second element
        from 0 to K-1.
    """
    if max_rep <= 0:
        raise ValueError("max_rep must be greater than 0.")
    if max_rep_strict is None and not np.isinf(max_rep):
        raise ValueError("if max_rep_strict is not set, then max_rep must be infinite.")
    # Make it so the axes we want to find the maximum along are the first ones...
    perm = list(range(len(x.shape)))
    for axis in reversed(axes):
        loc = perm.index(axis)
        perm = [axis] + perm[:loc] + perm[loc + 1 :]
    # ... by transposing like this.
    x = x.transpose(perm)
    shape = x.shape
    # Number of elements in those axes.
    axes_size = reduce(lambda a, b: a * b, shape[: len(axes)], 1)
    x = x.reshape([axes_size, -1])  # Flatten
    # Array containing for every sample the indices sorted by the values at that index,
    # largest values first.
    indices = np.argsort(-x, axis=0)
    result = indices[0].copy()  # The indices of the maximum values.
    counts = np.zeros(len(indices), dtype=int)
    unique_values, unique_counts = np.unique(result, return_counts=True)
    counts[unique_values] = unique_counts
    # For every i we look at every index and see if it is repeated too often, if so we
    # take the ith biggest index instead. We then update the counts and repeat this
    # with a bigger i, until we have an i where no index is repeated too often, then
    # we stop.
    for i in range(1, len(indices) + (0 if max_rep_strict else 1)):
        # Get the actual maxima and then sort them, order contains indices that sort
        # maxima with smallest maximum first.
        order = np.argsort(x[result, range(len(result))])
        result_in_order = result[order]  # The indices of maxima, but sorted
        current_counts = counts.copy()
        changed = False
        for j in range(len(order)):
            # Index of jth maximum (lowest maxs first) in orginal array.
            value = result_in_order[j]
            if current_counts[value] > max_rep:  # If there are more counts for this idx
                pos = order[j]  # position of jth max in result
                # Use the ith largest max instead. So for i=1, don't use largest max,
                # but instead second largest and so on. new_value is not the actual
                # maximum, but its index.
                new_value = indices[i % len(indices)][pos]
                result[pos] = new_value  # Update which maximum should be in results
                # Update the respective counts.
                current_counts[value] -= 1
                counts[value] -= 1
                counts[new_value] += 1
                changed = True
        if not changed:
            break
    result = result.reshape(shape[len(axes) :])
    # Returns a tuple of the indexes of maximal values.
    return np.unravel_index(result, shape[: len(axes)])


@wrap_objective()
def l2_objective(layer_name, coefficient, batch=None):
    """L2 norm of specified layer, multiplied by the given coeff."""

    @handle_batch(batch)
    def inner(model):
        return coefficient * th.sqrt(th.sum(model(layer_name) ** 2))

    return inner


class LayerNMF:
    acts_reduced: np.ndarray

    def __init__(
        self,
        model: th.nn.Module,
        layer_name: str,
        model_inputs_preprocess: th.Tensor,
        model_inputs_full: Optional[th.Tensor] = None,
        features: Optional[int] = 10,
        *,
        attr_layer_name: Optional[str] = None,
        attr_opts: Dict[str, int] = {"integrate_steps": 10},
        activation_fn: Optional[str] = None,
    ):
        """Use Non-negative matrix factorization dimensionality reduction to then do
        visualization.


        Args:
            model: The PyTorch model to analyze. Can be reward net or policy net.
            layer_name: The name of the layer to analyze.
            model_inputs_preprocess:
                Tensor of inputs to the model to use for preprocessing for
                visualization. Used for dimensionality reduction, if applicable, and
                for determining shape of activations in the model.
            model_inputs_full:
                Tensor containing all data to use for dataset visualization. If None,
                use model_inputs_preprocess.
            features:
                Number of features to use in NMF. None performs no dimensionality
                reduction.
            attr_layer_name:
                Name of the layer of attributions to apply NMF to. If None, apply NMF to
                activations.
            attr_opts:
               Options passed to get_grad_or_attr() from
               reward_preprocessing.vis.attribution.
            activation_fn:
                An optional additional activation function to apply to the activations.
                Sometimes "activations" in the model did not go through an actual
                activation function, e.g. the output of a reward net. If this
                activation function is specified, we will apply the respective function
                before doing NMF. This is especially important if activations (such as
                reward output) can have negative values.
        """
        if attr_layer_name is not None:
            logging.warning(
                "Doing gradient-based feature visualization on attributions might not "
                "work 100% correctly yet."
            )

        self.model = model
        self.layer_name = layer_name
        self.model_inputs_preprocess = model_inputs_preprocess
        self.model_inputs_full = model_inputs_full
        if self.model_inputs_full is None:
            self.model_inputs_full = model_inputs_preprocess
        self.features = features
        self.pad_h = 0
        self.pad_w = 0
        self.padded_obses = self.model_inputs_full
        # We want to reduce dim 1, which is the convention for channel dim in *lucent*
        # (not lucid). We assume different channels correspond to different features
        # for benefits of interpretability (as the do e.g. in the rl vision distill
        # paper). This is used for ChannelReducers.
        reduction_dim = 1
        if self.features is None:
            self.reducer = None
        else:
            # Dimensionality reduction using NMF.
            self.reducer = ChannelReducer(features, reduction_dim=reduction_dim)
        activations = get_activations(model, layer_name, model_inputs_preprocess)

        # Apply activation function if specified.
        if activation_fn == "sigmoid":
            activations = th.sigmoid(activations)
        elif activation_fn == "relu":
            relu_func = th.nn.ReLU()
            activations = relu_func(activations)
        elif activation_fn is not None:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        # From this point on activations should be non-negative.
        if activations.min() < 0:
            logging.warning(
                f"LayerNMF: activations for layer {layer_name} have negative values."
            )

        self.patch_h = self.model_inputs_full.shape[2] / activations.shape[2]
        self.patch_w = self.model_inputs_full.shape[3] / activations.shape[3]

        # From here on activations should be numpy array and not pytorch tensor anymore.
        activations = activations.detach().cpu().numpy()

        if self.reducer is None:  # No dimensionality reduction.
            # Activations are only used for dim reduction and to determine the shape
            # of the features. The former is compatible between torch and numpy (both
            # support .shape), so calling .numpy() is not really necessary. However,
            # for consistency we do it here. Consequently, self.acts_reduced is always
            # a numpy array.
            self.acts_reduced = activations
            self.channel_dirs = np.eye(self.acts_reduced.shape[1])
            self.transform = lambda acts: acts.copy()
            self.inverse_transform = lambda acts: acts.copy()
        else:  # Perform NMF dimensionality reduction.
            if attr_layer_name is None:
                # Perform the NMF reduction and return reduced tensor.
                self.acts_reduced = self.reducer.fit_transform(activations)
            else:
                attrs = (
                    get_attr(
                        model,
                        attr_layer_name,
                        layer_name,
                        model_inputs_preprocess,
                        **attr_opts,
                    )
                    .detach()
                    .numpy()
                )
                attrs_signed = np.concatenate(
                    [np.maximum(0, attrs), np.maximum(0, -attrs)], axis=0
                )
                # Use torch tensors so it is the same data type as 'activations', which
                # is a torch tensor.
                self.reducer.fit(th.tensor(attrs_signed))
                self.acts_reduced = self.reducer.transform(activations)

            self.channel_dirs = self.reducer._reducer.components_
            self.transform = lambda acts: self.reducer.transform(acts.cpu())
            self.inverse_transform = lambda acts_r: ChannelReducer._apply_flat(
                self.reducer._reducer.inverse_transform,
                acts_r,
                reduction_dim=reduction_dim,
            )
        # Transform into torch tensor instead of numpy array, because this is expected
        # later on.
        self.channel_dirs = th.tensor(self.channel_dirs).to(
            self.model_inputs_full.device
        )

    def vis_traditional(
        self,
        feature_list=None,
        *,
        transforms: List[Callable[[th.Tensor], th.Tensor]] = [transform.jitter(2)],
        l2_coeff: float = 0.0,
        l2_layer_name: Optional[str] = None,
        param_f: Optional[
            Callable[[], Tuple[th.Tensor, Callable[[], th.Tensor]]]
        ] = None,
    ) -> np.ndarray:
        if feature_list is None:
            # Feature dim is at index 1
            feature_list = list(range(self.acts_reduced.shape[1]))
        try:
            feature_list = list(feature_list)
        except TypeError:
            feature_list = [feature_list]

        obj = sum(
            # Original with cosine similarity (for if we go back to interpreting neuron
            # directions in intermediate layers_:
            # [
            #     objectives_rfi.direction_neuron_dim_agnostic(
            #         self.layer_name, self.channel_dirs[feature], batch=feature
            #     )
            #     for feature in feature_list
            # ]
            # New:
            # Sum up all objectives such that we simultaneously optimize for all.
            # Each objective maximizes the output for one of the activations (in this
            # case equivalent to the reward for the respective actions, or overall
            # reward if we don't differentiate between actions) and depends only on the
            # input at that same index.
            # In other words, each input maximizes its respective activation.
            [
                objectives_rfi.max_index_1d(self.layer_name, feature, batch=feature)
                for feature in feature_list
            ]
        )
        if l2_coeff != 0.0:
            if l2_layer_name is None:
                raise ValueError(
                    "l2_layer_name must be specified if l2_coeff is non-zero"
                )
            obj -= l2_objective(l2_layer_name, l2_coeff)
        input_shape = tuple(self.model_inputs_preprocess.shape[1:])

        if param_f is None:

            def param_f():
                return param.image(
                    channels=input_shape[0],
                    h=input_shape[1],
                    w=input_shape[2],
                    batch=len(feature_list),
                )

        logging.info(f"Performing vis_traditional with transforms: {transforms}")

        return render.render_vis(
            self.model,
            obj,
            param_f=param_f,
            transforms=transforms,
            # Don't use this preprocessing, this uses some default normalization for
            # ImageNet torchvision models, which of course assumes 3 channels and square
            # images as inputs
            preprocess=False,
            # This makes it so input is passed through the model at least once, which
            # is necessary to get the feature activations.
            verbose=True,
            # We work with fixed image sizes since our models do not accept arbitrary
            # sizes. If this is set to None (the default), the image will be upsampled
            # to (3, 224, 224).
            # Image size should be the spatial size (excluding channels).
            fixed_image_size=input_shape[1:],
            # Disable because our inputs ("images") are not actually images but
            # multidimensional tensors.
            show_image=False,
        )[-1]

    def pad_obses(self, *, expand_mult=1):
        pad_h = np.ceil(self.patch_h * expand_mult).astype(int)
        pad_w = np.ceil(self.patch_w * expand_mult).astype(int)
        if pad_h > self.pad_h or pad_w > self.pad_w:
            self.pad_h = pad_h
            self.pad_w = pad_w
            # The image shape we want to pad to (only 2d i.e. height and width).
            self.padded_obses = (
                np.indices(
                    (
                        self.model_inputs_full.shape[2] + self.pad_h * 2,
                        self.model_inputs_full.shape[3] + self.pad_w * 2,
                    )
                ).sum(axis=0)
                % 2
            )  # Checkered pattern.
            self.padded_obses = self.padded_obses * 0.25 + 0.75  # Adjust color.
            self.padded_obses = self.padded_obses.astype(
                self.model_inputs_full.detach().cpu().numpy().dtype
            )
            # Add dims for batch and channel.
            self.padded_obses = self.padded_obses[None, None, ...]
            # Repeat for correct number of images.
            self.padded_obses = self.padded_obses.repeat(
                self.model_inputs_full.shape[0], axis=0
            )
            # Repeat channel dimension.
            num_channels = self.model_inputs_full.shape[1]
            self.padded_obses = self.padded_obses.repeat(num_channels, axis=1)
            self.padded_obses[
                :, :, self.pad_h : -self.pad_h, self.pad_w : -self.pad_w
            ] = (self.model_inputs_full.detach().cpu().numpy())

    def get_patch(self, obs_index, pos_h, pos_w, *, expand_mult=1):
        left_h = self.pad_h + (pos_h - 0.5 * expand_mult) * self.patch_h
        right_h = self.pad_h + (pos_h + 0.5 * expand_mult) * self.patch_h
        left_w = self.pad_w + (pos_w - 0.5 * expand_mult) * self.patch_w
        right_w = self.pad_w + (pos_w + 0.5 * expand_mult) * self.patch_w
        slice_h = slice(int(round(left_h)), int(round(right_h)))
        slice_w = slice(int(round(left_w)), int(round(right_w)))
        return self.padded_obses[obs_index, :, slice_h, slice_w]

    def vis_dataset(
        self,
        feature: Union[int, List[int]],
        *,
        subdiv_mult=1,
        expand_mult=1,
        top_frac: float = 0.1,
    ):
        """Visualize a dataset of patches that maximize a given feature.

        Args:
            feature: The feature to visualize. Can be an integer or a list of integers.
        """
        logging.warning(
            "Dataset-based feature visualization seems to still have some problems."
        )
        acts_h, acts_w = self.acts_reduced.shape[2:]
        zoom_h = subdiv_mult - (subdiv_mult - 1) / (acts_h + 2)
        zoom_w = subdiv_mult - (subdiv_mult - 1) / (acts_w + 2)
        acts_subdiv = self.acts_reduced[..., feature]
        acts_subdiv = np.pad(acts_subdiv, [(0, 0), (1, 1), (1, 1)], mode="edge")
        acts_subdiv = nd.zoom(acts_subdiv, [1, zoom_h, zoom_w], order=1, mode="nearest")
        acts_subdiv = acts_subdiv[:, 1:-1, 1:-1]
        if acts_subdiv.size == 0:
            raise RuntimeError(
                f"subdiv_mult of {subdiv_mult} too small for "
                f"{self.acts_reduced.shape[1]}x{self.acts_reduced.shape[2]} "
                "activations"
            )
        poses = np.indices((acts_h + 2, acts_w + 2)).transpose((1, 2, 0))
        poses = nd.zoom(
            poses.astype(float), [zoom_h, zoom_w, 1], order=1, mode="nearest"
        )
        poses = poses[1:-1, 1:-1, :] - 0.5
        with np.errstate(divide="ignore"):
            max_rep = np.ceil(
                np.divide(
                    acts_subdiv.shape[1] * acts_subdiv.shape[2],
                    acts_subdiv.shape[0] * top_frac,
                )
            )
        obs_indices = argmax_nd(
            acts_subdiv, axes=[0], max_rep=max_rep, max_rep_strict=False
        )[0]
        self.pad_obses(expand_mult=expand_mult)
        patches = []
        patch_acts = np.zeros(obs_indices.shape)
        for i in range(obs_indices.shape[0]):
            patches.append([])
            for j in range(obs_indices.shape[1]):
                obs_index = obs_indices[i, j]
                pos_h, pos_w = poses[i, j]
                patch = self.get_patch(obs_index, pos_h, pos_w, expand_mult=expand_mult)
                patches[i].append(patch)
                patch_acts[i, j] = acts_subdiv[obs_index, i, j]
        patch_acts_max = patch_acts.max()
        opacities = patch_acts / (1 if patch_acts_max == 0 else patch_acts_max)
        for i in range(obs_indices.shape[0]):
            for j in range(obs_indices.shape[1]):
                opacity = opacities[i, j][None, None, None]
                opacity = opacity.repeat(patches[i][j].shape[0], axis=0)
                opacity = opacity.repeat(patches[i][j].shape[1], axis=1)
                patches[i][j] = np.concatenate([patches[i][j], opacity], axis=-1)
        return (
            np.concatenate(
                [np.concatenate(patches[i], axis=1) for i in range(len(patches))],
                axis=0,
            ),
            obs_indices.tolist(),
        )

    def vis_dataset_thumbnail(
        self,
        feature: Union[int, List[int]],
        *,
        num_mult: int = 1,
        expand_mult: int = 1,
        max_rep: Optional[Union[int, float]] = None,
    ):
        """Visualize a dataset of patches that maximize a given feature.

        Args:
            feature: The feature to visualize. Can be an integer or a list of integers.
            num_mult: Height and width of the grid of thumbnails.
            expand_mult: Multiplier for the size of the thumbnails.
            max_rep: Maximum number of times the same observation can appear.
        """
        logging.warning(
            "Dataset-based feature visualization seems to still have some problems."
        )
        if max_rep is None:
            max_rep = num_mult
        if self.acts_reduced.shape[0] < num_mult**2:
            raise RuntimeError(
                f"At least {num_mult ** 2} observations are required to produce"
                " a thumbnail visualization."
            )
        # Feature dim = channel dim = second dim
        acts_feature = self.acts_reduced[:, feature]
        pos_indices = argmax_nd(
            acts_feature, axes=[1, 2], max_rep=max_rep, max_rep_strict=True
        )
        # The actual maximum values of the activations, according to max_rep setting.
        acts_single = acts_feature[
            range(acts_feature.shape[0]), pos_indices[0], pos_indices[1]
        ]
        # Sort the activations in descending order and take the num_mult**2 strongest
        # activations.
        obs_indices = np.argsort(-acts_single, axis=0)[: num_mult**2]

        # Coordinates of the strongest activation in each observation.
        coords = np.array(list(zip(*pos_indices)), dtype=[("h", int), ("w", int)])[
            obs_indices
        ]
        # Sort by indices.
        indices_order = np.argsort(coords, axis=0, order=("h", "w"))
        # Make into num_mult x num_mult grid.
        indices_order = indices_order.reshape((num_mult, num_mult))
        # Also order in the second dimension.
        for i in range(num_mult):
            indices_order[i] = indices_order[i][
                np.argsort(coords[indices_order[i]], axis=0, order="w")
            ]
        # obs_indices now contains the indices of the observations in the order as
        # ordered above.
        obs_indices = obs_indices[indices_order]
        poses = np.array(pos_indices).transpose()[obs_indices] + 0.5
        self.pad_obses(expand_mult=expand_mult)
        patches = []
        patch_acts = np.zeros((num_mult, num_mult))
        patch_shapes = []
        for i in range(num_mult):
            patches.append([])
            for j in range(num_mult):
                obs_index = obs_indices[i, j]
                pos_h, pos_w = poses[i, j]
                patch = self.get_patch(obs_index, pos_h, pos_w, expand_mult=expand_mult)
                patches[i].append(patch)
                patch_acts[i, j] = acts_single[obs_index]
                patch_shapes.append(patch.shape)
        patch_acts_max = patch_acts.max()
        opacities = patch_acts / (1 if patch_acts_max == 0 else patch_acts_max)
        patch_min_h = np.array([s[1] for s in patch_shapes]).min()
        patch_min_w = np.array([s[2] for s in patch_shapes]).min()
        for i in range(num_mult):
            for j in range(num_mult):
                opacity = opacities[i, j][None, None, None]
                opacity = opacity.repeat(patches[i][j].shape[1], axis=1)
                opacity = opacity.repeat(patches[i][j].shape[2], axis=2)
                patches[i][j] = np.concatenate([patches[i][j], opacity], axis=0)
                patches[i][j] = patches[i][j][:, :patch_min_h, :patch_min_w]
        # Concat first along y dim then along x dim to have 1 big image that is a grid
        # of the smaller images.
        return (
            np.concatenate(
                [np.concatenate(patches[i], axis=2) for i in range(len(patches))],
                axis=1,
            ),
            obs_indices.tolist(),
        )

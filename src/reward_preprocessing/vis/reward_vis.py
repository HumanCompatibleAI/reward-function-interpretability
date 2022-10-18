"""Port of lucid.scratch.rl_util to PyTorch. APL2.0 licensed."""
from functools import reduce
import logging
from typing import Optional

import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
import numpy as np
import scipy.ndimage as nd
import torch as th

from reward_preprocessing.ext.channel_reducer import ChannelReducer
from reward_preprocessing.vis.attribution import get_acts, get_attr


def argmax_nd(x, axes, *, max_rep=np.inf, max_rep_strict=None):
    assert max_rep > 0
    assert np.isinf(max_rep) or max_rep_strict is not None
    perm = list(range(len(x.shape)))
    for axis in reversed(axes):
        loc = perm.index(axis)
        perm = [axis] + perm[:loc] + perm[loc + 1 :]
    x = x.transpose(perm)
    shape = x.shape
    axes_size = reduce(lambda a, b: a * b, shape[: len(axes)], 1)
    x = x.reshape([axes_size, -1])
    indices = np.argsort(-x, axis=0)
    result = indices[0].copy()
    counts = np.zeros(len(indices), dtype=int)
    unique_values, unique_counts = np.unique(result, return_counts=True)
    counts[unique_values] = unique_counts
    for i in range(1, len(indices) + (0 if max_rep_strict else 1)):
        order = np.argsort(x[result, range(len(result))])
        result_in_order = result[order]
        current_counts = counts.copy()
        changed = False
        for j in range(len(order)):
            value = result_in_order[j]
            if current_counts[value] > max_rep:
                pos = order[j]
                new_value = indices[i % len(indices)][pos]
                result[pos] = new_value
                current_counts[value] -= 1
                counts[value] -= 1
                counts[new_value] += 1
                changed = True
        if not changed:
            break
    result = result.reshape(shape[len(axes) :])
    return np.unravel_index(result, shape[: len(axes)])


class LayerNMF:
    def __init__(
        self,
        model,
        layer_name,
        obses,
        obses_full=None,
        features: Optional[int] = 10,
        *,
        attr_layer_name: Optional[str] = None,
        attr_opts={"integrate_steps": 10},
        activation_fn: Optional[str] = None,
    ):
        """Use Non-negative matrix factorization dimensionality reduction to then do
        visualization.


        Args:
            model: The PyTorch model to analyze. Can be reward net or policy net.
            layer_name: The name of the layer to analyze.
            obses: Dataset of observations to analyze.
            obses_full:
            features:
                Number of features to use in NMF. None performs no dimensionality
                reduction.
            attr_layer_name:
                Name of the layer of attributions to apply NMF to. If None, apply NMF to
                activations.
            attr_opts:
            activation_fn:
                An optional additional activation function to apply to the activations.
                Sometimes "activations" in the model did not go through an actual
                activation function, e.g. the output of a reward net. If this
                activation function is specified, we will apply the respective function
                before doing NMF. This is especially important if activations (such as
                reward outpu) can have negative values.
        """
        self.model = model
        self.layer_name = layer_name
        self.obses = obses
        self.obses_full = obses_full
        if self.obses_full is None:
            self.obses_full = obses
        self.features = features
        self.pad_h = 0
        self.pad_w = 0
        self.padded_obses = self.obses_full

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
        activations = get_acts(model, layer_name, obses)

        # Apply activation function if specified.
        if activation_fn == "sigmoid":
            activations = th.sigmoid(activations)
        elif activation_fn is not None:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        # From this point on activations should be non-negative.
        if activations.min() < 0:
            logging.warning(
                f"LayerNMF: activations for layer {layer_name} have negative values."
            )

        self.patch_h = self.obses_full.shape[1] / activations.shape[1]
        self.patch_w = self.obses_full.shape[2] / activations.shape[2]
        if self.reducer is None:  # No dimensionality reduction.
            self.acts_reduced = activations
            self.channel_dirs = np.eye(self.acts_reduced.shape[-1])
            self.transform = lambda acts: acts.copy()
            self.inverse_transform = lambda acts: acts.copy()
        else:  # Perform NMF dimensionality reduction
            if attr_layer_name is None:
                # Perform the NMF reduction and return reduced tensor.
                self.acts_reduced = self.reducer.fit_transform(activations)
            else:
                attrs = get_attr(model, attr_layer_name, layer_name, obses, **attr_opts)
                attrs_signed = np.concatenate(
                    [np.maximum(0, attrs), np.maximum(0, -attrs)], axis=0
                )
                self.reducer.fit(attrs_signed)
                self.acts_reduced = self.reducer.transform(activations)
            self.channel_dirs = self.reducer._reducer.components_
            self.transform = lambda acts: self.reducer.transform(acts)
            self.inverse_transform = lambda acts_r: ChannelReducer._apply_flat(
                self.reducer._reducer.inverse_transform,
                acts_r,
                reduction_dim=reduction_dim,
            )
        # Transform into torch tensor instead of numpy array, because this is expected
        # later on.
        self.channel_dirs = th.tensor(self.channel_dirs)

    def vis_traditional(
        self,
        feature_list=None,
        *,
        transforms=[transform.jitter(2)],
        l2_coeff=0.0,
        l2_layer_name=None,
    ):
        if feature_list is None:
            feature_list = list(range(self.acts_reduced.shape[-1]))
        try:
            feature_list = list(feature_list)
        except TypeError:
            feature_list = [feature_list]

        obj = sum(
            [
                objectives.direction_neuron(
                    self.layer_name, self.channel_dirs[feature], batch=feature
                )
                for feature in feature_list
            ]
        )
        if l2_coeff != 0.0:
            assert (
                l2_layer_name is not None
            ), "l2_layer_name must be specified if l2_coeff is non-zero"
            obj -= objectives.L2(l2_layer_name) * l2_coeff
        param_f = lambda: param.image(64, batch=len(feature_list))
        return render.render_vis(
            self.model,
            obj,
            param_f=param_f,
            transforms=transforms,
            # To fix order of transforms.
            # TODO: Should we enable preprocess here?
            preprocess=False,
            # This makes it so input is passed through the model at least ones, which
            # is necessary to get the feature activations.
            verbose=True,
            # We work with images of size 64, the model does not accept arbitrary sizes.
            fixed_image_size=64,
        )[-1]

    def pad_obses(self, *, expand_mult=1):
        pad_h = np.ceil(self.patch_h * expand_mult).astype(int)
        pad_w = np.ceil(self.patch_w * expand_mult).astype(int)
        if pad_h > self.pad_h or pad_w > self.pad_w:
            self.pad_h = pad_h
            self.pad_w = pad_w
            self.padded_obses = (
                np.indices(
                    (
                        self.obses_full.shape[1] + self.pad_h * 2,
                        self.obses_full.shape[2] + self.pad_w * 2,
                    )
                ).sum(axis=0)
                % 2
            )
            self.padded_obses = self.padded_obses * 0.25 + 0.75
            self.padded_obses = self.padded_obses.astype(self.obses_full.dtype)
            self.padded_obses = self.padded_obses[None, ..., None]
            self.padded_obses = self.padded_obses.repeat(
                self.obses_full.shape[0], axis=0
            )
            self.padded_obses = self.padded_obses.repeat(3, axis=-1)
            self.padded_obses[
                :, self.pad_h : -self.pad_h, self.pad_w : -self.pad_w, :
            ] = self.obses_full

    def get_patch(self, obs_index, pos_h, pos_w, *, expand_mult=1):
        left_h = self.pad_h + (pos_h - 0.5 * expand_mult) * self.patch_h
        right_h = self.pad_h + (pos_h + 0.5 * expand_mult) * self.patch_h
        left_w = self.pad_w + (pos_w - 0.5 * expand_mult) * self.patch_w
        right_w = self.pad_w + (pos_w + 0.5 * expand_mult) * self.patch_w
        slice_h = slice(int(round(left_h)), int(round(right_h)))
        slice_w = slice(int(round(left_w)), int(round(right_w)))
        return self.padded_obses[obs_index, slice_h, slice_w]

    def vis_dataset(self, feature, *, subdiv_mult=1, expand_mult=1, top_frac=0.1):
        """Visualize a dataset of patches that maximize a given feature.

        Args:
            feature: The feature to visualize. Can be an integer or a list of integers.
        """
        acts_h, acts_w = self.acts_reduced.shape[1:3]
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
        self, feature, *, num_mult=1, expand_mult=1, max_rep=None
    ):
        """Visualize a dataset of patches that maximize a given feature.

        Args:
            feature: The feature to visualize. Can be an integer or a list of integers.
            num_mult: Height and width of the grid of thumbnails.
            expand_mult: Multiplier for the size of the thumbnails.
        """
        if max_rep is None:
            max_rep = num_mult
        if self.acts_reduced.shape[0] < num_mult**2:
            raise RuntimeError(
                f"At least {num_mult ** 2} observations are required to produce"
                " a thumbnail visualization."
            )
        acts_feature = self.acts_reduced[..., feature]
        pos_indices = argmax_nd(
            acts_feature, axes=[1, 2], max_rep=max_rep, max_rep_strict=True
        )
        acts_single = acts_feature[
            range(acts_feature.shape[0]), pos_indices[0], pos_indices[1]
        ]
        obs_indices = np.argsort(-acts_single, axis=0)[: num_mult**2]
        coords = np.array(list(zip(*pos_indices)), dtype=[("h", int), ("w", int)])[
            obs_indices
        ]
        indices_order = np.argsort(coords, axis=0, order=("h", "w"))
        indices_order = indices_order.reshape((num_mult, num_mult))
        for i in range(num_mult):
            indices_order[i] = indices_order[i][
                np.argsort(coords[indices_order[i]], axis=0, order="w")
            ]
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
        patch_min_h = np.array([s[0] for s in patch_shapes]).min()
        patch_min_w = np.array([s[1] for s in patch_shapes]).min()
        for i in range(num_mult):
            for j in range(num_mult):
                opacity = opacities[i, j][None, None, None]
                opacity = opacity.repeat(patches[i][j].shape[0], axis=0)
                opacity = opacity.repeat(patches[i][j].shape[1], axis=1)
                patches[i][j] = np.concatenate([patches[i][j], opacity], axis=-1)
                patches[i][j] = patches[i][j][:patch_min_h, :patch_min_w]
        return (
            np.concatenate(
                [np.concatenate(patches[i], axis=1) for i in range(len(patches))],
                axis=0,
            ),
            obs_indices.tolist(),
        )


# def rescale_opacity(
#     images, min_opacity=15 / 255, opaque_frac=0.1, max_scale=10, keep_zeros=False
# ):
#     images_orig = images
#     images = images_orig.copy()
#     opacities_flat = images[..., 3].reshape(
#         images.shape[:-3] + (images.shape[-3] * images.shape[-2],)
#     )
#     opaque_threshold = np.percentile(opacities_flat, (1 - opaque_frac) * 100, axis=-1)
#     opaque_threshold = np.maximum(
#         opaque_threshold, np.amax(opacities_flat, axis=-1) / max_scale
#     )[..., None, None]
#     opaque_threshold[opaque_threshold == 0] = 1
#     images[..., 3] = images[..., 3] * (1 - min_opacity) / opaque_threshold
#     images[..., 3] = np.minimum(1, min_opacity + images[..., 3])
#     if keep_zeros:
#         images[..., 3][images_orig[..., 3] == 0] = 0
#     return images

"""Objectives that extend the objectives available in lucent.optvis.objectives"""
from typing import Optional

from lucent.optvis.objectives import handle_batch, wrap_objective
from lucent.optvis.objectives_util import _extract_act_pos
import torch as th


@wrap_objective()
def max_index_1d(layer: str, i: int, batch: Optional[int] = None):
    """Maximize the value at a specific index in a 1D tensor."""

    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        # This is (batch_size, n), we want to maximize the ith element of each batch.
        return -layer_t[:, i].mean()

    return inner


@wrap_objective()
def direction_neuron_dim_agnostic(layer, direction, x=None, y=None, batch=None):
    """The lucent direction neuron objective, modified to allow 2-dimensional
    activations.
    For more info see lucent.optvis.objectives.direction_neuron.
    """

    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        if len(layer_t.shape) == 2:
            layer_t = layer_t.unsqueeze(-1).unsqueeze(-1)
        assert len(layer_t.shape) == 4, (
            "activations must be 4 dimensional, after intervention, dim of "
            f"activations is {layer_t.shape}."
        )
        layer_t = _extract_act_pos(layer_t, x, y)
        return -th.nn.CosineSimilarity(dim=1)(
            direction.reshape((1, -1, 1, 1)), layer_t
        ).mean()

    return inner

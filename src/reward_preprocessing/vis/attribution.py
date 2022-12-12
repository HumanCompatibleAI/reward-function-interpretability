"""Port of lucid.scratch.attribution to PyTorch. APL2.0 licensed."""
import lucent.optvis.render as render
import numpy as np
import torch as th
from torch import nn


def get_activations(
    model: nn.Module, layer_name: str, model_inputs: th.Tensor
) -> th.Tensor:
    """Get the activations of a layer in a model for a batch of inputs to the model."""
    hook = render.hook_model(model, model_inputs)

    # Perform forward pass through input to hook activations with a reasonable batch
    # size. Generally the number of inputs would be in the thousands, so we don't want
    # to run the entire batch through the model at once.
    batch_size = 128
    t_acts = []
    for i in range(0, len(model_inputs), batch_size):
        model(model_inputs[i : i + batch_size])

        # Get activations at layer.
        act_batch = hook(layer_name)
        t_acts.append(act_batch.detach())
        model.zero_grad()

    t_acts = th.cat(t_acts, dim=0)
    assert t_acts.shape[0] == len(model_inputs)

    # Reward activations might be 2 dimensional (scalar + batch dimension) e.g.
    # for linear layers. In this case we unsqueeze.
    if len(t_acts.shape) == 2:
        t_acts = t_acts.unsqueeze(-1).unsqueeze(-1)
    assert (
        len(t_acts.shape) >= 4
    ), "activations should be at least 3 dimensional plus a batch dimension"
    return t_acts.detach()


def default_score_fn(t):
    return th.sum(t, dim=list(range(1, len(t.shape))))


def get_grad_or_attr(
    model,
    layer_name,
    prev_layer_name,
    model_inputs,
    *,
    act_dir=None,
    activation_positions=None,
    score_fn=default_score_fn,
    grad_or_attr,
    override=None,
    integrate_steps=1,
):
    # This is a WIP port to PyTorch. The following parameters are not yet implemented
    # and we therefore assert that they are not used.
    # Currently we don't need them, but maybe we want to implement them in the future
    # for closer parity with the tf code. If they are used accidentally the exception
    # will be thrown. TODO: Implement or remove.
    if activation_positions is not None:
        raise NotImplementedError("act_poses not implemented. See comment above.")
    if override is not None:
        raise NotImplementedError("override not implemented. See comment above.")
    model_inputs = th.from_numpy(model_inputs.astype(np.float32))
    hook = render.hook_model(model, model_inputs)
    # Run through model once to generate feature maps
    model(model_inputs)
    # The activations of the layer we want to compute the attribution for.
    t_activations = hook(layer_name)
    if prev_layer_name is None:
        t_activations_prev = model_inputs
    else:
        # The activations of the previous layer.
        t_activations_prev = hook(prev_layer_name)
    if act_dir is not None:
        t_activations = act_dir[None, None, None] * t_activations
    t_scores = score_fn(t_activations)
    assert len(t_scores.shape) >= 1, "score_fn should not reduce the batch dim"
    t_score = th.sum(t_scores)
    if integrate_steps > 1:
        # This was the original tf 1 code. Keeping this around for transparency.
        # acts_prev = t_acts_prev.eval()
        # grad = (
        #     sum(
        #         [
        #             t_grad.eval(feed_dict={t_acts_prev: acts_prev * alpha})
        #             for alpha in np.linspace(0, 1, integrate_steps + 1)[1:]
        #         ]
        #     )
        #     / integrate_steps
        # )
        # The feed_dict argument allows the caller to override the value of tensors in
        # the graph.
        # This is how I wrote it in PyTorch (including the for loop):
        acts_prev = t_activations_prev
        grad_list = []
        for alpha in np.linspace(0, 1, integrate_steps + 1)[1:]:
            t_score = th.sum(t_scores)
            # This is what they did in tensorflow, equivalent:
            # grad = th.autograd.grad(t_score, [acts_prev * alpha])[0]
            # However, this doesn't work in pytorch because now PyTorch will try to
            # compute the gradient wrt alpha, which is 0 since alpha was not involved
            # in the input / isn't part of the graph. However, arithmetically, one can
            # pull out the scalar like this:
            # d t_scr / d (a_prv * alpha) = d t_scr / (alpha * d a_prv)
            # Therefore the following:
            grad = th.autograd.grad(t_score, [acts_prev], retain_graph=True)[0] / alpha
            grad_list.append(grad)
        grad = np.sum(grad_list) / integrate_steps
    else:
        acts_prev = None
        # Gradients from score to either inputs or the features at specified layer
        grad = th.autograd.grad(t_score, [t_activations_prev])[0]
    if grad_or_attr == "grad":
        return grad
    elif grad_or_attr == "attr":
        if acts_prev is None:
            acts_prev = t_activations_prev
        return acts_prev * grad
    else:
        raise NotImplementedError


def get_attr(model, layer_name, prev_layer_name, obses, **kwargs):
    kwargs["grad_or_attr"] = "attr"
    return get_grad_or_attr(model, layer_name, prev_layer_name, obses, **kwargs)

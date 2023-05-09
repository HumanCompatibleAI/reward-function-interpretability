import copy
import dataclasses
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import PIL
from PIL import Image
from imitation.data import rollout, types
from imitation.rewards.reward_nets import RewardNet
from imitation.util.logger import HierarchicalLogger
import numpy as np
import torch as th
from torch import nn as nn
from torch.utils import data as torch_data
import vegans.utils
import wandb


def make_transition_to_tensor(num_acts):
    """Produces a function that takes a transition, produces a tensor.

    For use as something to 'map over' a torch dataset of transitions. Assumes
    observations are (h,w,c)-formatted images, actions are discrete.
    Output tensor will have shape (2*c + num_acts, h, w).
    Order is (obs, act, next_obs).

    Args:
        num_acts: Number of discrete actions. Necessary because actions are
            saved as integers rather than one-hot vectors.
    """

    def transition_to_tensor(transition):
        obs = transition["obs"]
        if np.issubdtype(obs.dtype, np.integer):
            obs = obs / 255.0
            # For floats we don't divide by 255.0. In that case we assume the
            # observation is already in the range [0, 1].
        act = int(transition["acts"])
        next_obs = transition["next_obs"]

        if np.issubdtype(next_obs.dtype, np.integer):
            next_obs = next_obs / 255.0

        transp_obs = np.transpose(obs, (2, 0, 1))
        obs_height = transp_obs.shape[1]
        obs_width = transp_obs.shape[2]
        act_one_hot = np.zeros(num_acts)
        act_one_hot[act] = 1
        act_one_hot = act_one_hot[:, None, None]
        boosted_act = np.broadcast_to(act_one_hot, (num_acts, obs_height, obs_width))
        transp_next_obs = np.transpose(next_obs, (2, 0, 1))
        assert transp_next_obs.shape[1] == obs_height
        assert transp_next_obs.shape[2] == obs_width
        tensor_transition = np.concatenate(
            [transp_obs, boosted_act, transp_next_obs],
            axis=0,
        )
        return tensor_transition

    return transition_to_tensor


class TransformedDataset(torch_data.Dataset):
    """Map a function over a torch dataset"""

    def __init__(self, base_dataset, func):
        super().__init__()
        self.base_dataset = base_dataset
        self.func = func

    def __getitem__(self, idx):
        base_item = self.base_dataset.__getitem__(idx)
        return self.func(base_item)

    def __len__(self):
        return self.base_dataset.__len__()


def rollouts_to_dataloader(
    rollouts_paths: Union[str, List[str]],
    num_acts: int,
    batch_size: int,
    n_trajectories: Optional[int] = None,
):
    """Take saved rollouts of a policy, and produce a dataloader of transitions.

    Assumes that observations are (h,w,c)-formatted images and that actions are
    discrete.

    Args:
        rollouts_paths: Path to rollouts saved via imitation script, or list of
            such paths.
        num_acts: Number of actions available to the agent (necessary because
            actions are saved as a number, not as a one-hot vector).
        batch_size: Int, size of batches that the dataloader serves. Note that
            a batch size of 2 will make the GAN algorithm think each batch is
            a (data, label) pair, which will mess up training.
        n_trajectories: If not None, limit number of trajectories to use.
    """
    if isinstance(rollouts_paths, list):
        rollouts = []
        for path in rollouts_paths:
            rollouts += types.load_with_rewards(path)
    else:
        rollouts = types.load_with_rewards(rollouts_paths)

    # Optionally limit the number of trajectories to use, similar to n_expert_demos in
    # imitation.scripts.common.demonstrations.
    if n_trajectories is not None:
        if len(rollouts) < n_trajectories:
            raise ValueError(
                f"Want to use n_trajectories={n_trajectories} trajectories, but only "
                f"{len(rollouts)} are available via {rollouts_paths}.",
            )
        rollouts = rollouts[:n_trajectories]

    flat_rollouts = rollout.flatten_trajectories_with_rew(rollouts)
    tensor_rollouts = TransformedDataset(
        flat_rollouts, make_transition_to_tensor(num_acts)
    )
    rollout_dataloader = torch_data.DataLoader(
        tensor_rollouts, shuffle=True, batch_size=batch_size
    )
    return rollout_dataloader


def visualize_samples(samples: np.ndarray, save_dir):
    """Visualize samples from a GAN.

    Saves obs and next obs as png files, and takes mean over height and width dimensions
    to turn act into a numpy array, before saving it.
    """
    for i, transition in enumerate(samples):
        s, act, s_ = ndarray_to_transition(transition)
        s = process_image_array(s)
        s_ = process_image_array(s_)
        s_img = PIL.Image.fromarray(s)
        s__img = PIL.Image.fromarray(s_)
        (Path(save_dir) / str(i)).mkdir()
        s_img.save(Path(save_dir) / str(i) / "first_obs.png")
        s__img.save(Path(save_dir) / str(i) / "second_obs.png")
        np.save(Path(save_dir) / str(i) / "act.npy", act)


def process_image_array(img: np.ndarray) -> np.ndarray:
    """Process a numpy array for feeding into PIL.Image.fromarray.

    Should already be in (h,w,c) format.
    """
    up_multiplied = img * 255
    clipped = np.clip(up_multiplied, 0, 255)
    cast = clipped.astype(np.uint8)
    return cast


def tensor_to_transition(
    trans_tens: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Turn a generated 'transition tensor' batch into a batch of bona fide
    transitions. Output observations will have channel dim last, actions will be
    output as one-hot vectors.
    Assumes input transition tensor has values between 0 and 1.
    """
    num_acts = trans_tens.size(1) - 6
    # process first observation
    obs_raw = trans_tens[:, 0:3, :, :]
    obs_proc = process_image_tensor(obs_raw)
    # process action
    act_raw = trans_tens[:, 3 : num_acts + 3, :, :]
    act_slim = th.mean(act_raw, dim=[2, 3])
    arg_max = th.argmax(act_slim, dim=1)
    act_proc = nn.functional.one_hot(arg_max, num_classes=num_acts)
    # process next observation
    next_obs_raw = trans_tens[:, num_acts + 3 : num_acts + 6, :, :]
    next_obs_proc = process_image_tensor(next_obs_raw)
    return obs_proc, act_proc, next_obs_proc


def ndarray_to_transition(
    np_trans: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a numpy transition tensor into three bona fide transitions."""
    if len(np_trans.shape) != 3:
        raise ValueError("ndarray_to_transition assumes input has shape of length 3")
    boosted_np_trans = np_trans[None, :, :, :]
    th_trans = th.from_numpy(boosted_np_trans)
    th_obs, th_act, th_next_obs = tensor_to_transition(th_trans)
    np_obs, np_act, np_next_obs = map(
        lambda th_result: th_result[0].detach().cpu().numpy(),
        (th_obs, th_act, th_next_obs),
    )
    return np_obs, np_act, np_next_obs


def process_image_tensor(obs: th.Tensor) -> th.Tensor:
    """Take a GAN image and processes it for use in a reward net."""
    clipped_obs = th.clamp(obs, 0, 1)
    transposed = th.permute(clipped_obs, (0, 2, 3, 1))
    return transposed


class TensorTransitionWrapper(nn.Module):
    """Wraps an imitation-style reward net such that it accepts transitions tensors.
    Dones will always be a batch of zeros.
    """

    def __init__(self, rew_net: RewardNet):
        """rew_net should be a reward net that takes in (obs, act, next_obs, done) as
        arguments."""
        super().__init__()
        self.rew_net = rew_net
        self.transition_tensor_identity_op = nn.Identity()

    def forward(self, transition_tensor: th.Tensor) -> th.Tensor:
        # Input data must be between 0 and 1 because that is what
        # tensor_to_transition expects.
        transition_tensor_ = self.transition_tensor_identity_op(transition_tensor)
        obs, act, next_obs = tensor_to_transition(transition_tensor_)

        dones = th.zeros_like(obs[:, 0])
        return self.rew_net(state=obs, action=act, next_state=next_obs, done=dones)


class RewardGeneratorCombo(nn.Module):
    """Composition of a generative model and a RewardNet.

    Assumes that the RewardNet normalizes observations to [0,1].
    """

    def __init__(self, rew_net: RewardNet, generator: nn.Module):
        super().__init__()
        self.rew_net = rew_net
        self.generator = generator

    def forward(self, latent_tens: th.Tensor):
        latent_vec = th.mean(latent_tens, dim=[2, 3])
        transition_tensor = self.generator(latent_vec)
        obs, action_vec, next_obs = tensor_to_transition(transition_tensor)
        done = th.zeros(action_vec.shape)
        return self.rew_net.forward(obs, action_vec, next_obs, done)


def log_img_wandb(
    img: Union[np.ndarray, PIL.Image.Image],
    caption: str,
    wandb_key: str,
    logger: HierarchicalLogger,
    scale: int = 1,
    step: Optional[int] = None,
) -> None:
    """Log np.ndarray as image or PIL image. Logs to wandb using given logger.
    If scale is provided, image will be scaled in both cases.

    Args:
        - arr: Array to turn into image, save.
        - caption: Caption to give the image.
        - wandb_key: Key to use for logging to wandb.
        - logger: Logger to use.
        - scale: Ratio by which to scale up the image in spatial dimensions.
        - step: Step for logging. If not provided, the logger dumping will be skipped.
            In that case logs will be dumped with the next dump().
    """
    if isinstance(img, np.ndarray):
        pil_img = array_to_image(img, scale)
    elif isinstance(img, PIL.Image.Image):
        pil_img = img.resize(
            # PIL expects tuple of (width, height), as opposed to numpy's
            # (height, width).
            size=(img.width * scale, img.height * scale),
            resample=Image.NEAREST,
        )
    else:
        raise ValueError(f"img must be np.ndarray or PIL.Image.Image, {type(img)=}")
    wb_img = wandb.Image(pil_img, caption=caption)
    logger.record(wandb_key, wb_img)
    if step is not None:
        logger.dump(step=step)


def array_to_image(arr: np.ndarray, scale: int) -> PIL.Image.Image:
    """Take numpy array on [0,1] scale with shape (h,w,c), return PIL image."""
    return Image.fromarray(np.uint8(arr * 255), mode="RGB").resize(
        # PIL expects tuple of (width, height), numpy's dimension 1 is width, and
        # dimension 0 height.
        size=(arr.shape[1] * scale, arr.shape[0] * scale),
        resample=Image.NEAREST,
    )


def save_loss_plots(losses, save_dir):
    """Save plots of generator/adversary losses over training."""
    fig, _ = vegans.utils.plot_losses(losses, show=False)
    fig.savefig(Path(save_dir) / "loss_fig.png")


@dataclasses.dataclass(frozen=True)
class DoubleInfoTransitionsWithRew(types.TransitionsWithRew):
    """A batch of obs-info_dict-act-next_obs-next_info_dict-done transitions."""

    next_infos: np.ndarray
    """Array of info dicts corresponding to the next_obs observation.

    Shape: (batch_size,).
    """

    def __post_init__(self):
        """Validate that next_infos has right shape"""
        if len(self.next_infos) != len(self.next_obs):
            raise ValueError(
                "next_infos and next_obs must have same number of timesteps: "
                + f"{len(self.next_infos)} != {len(self.next_obs)}",
            )


def flatten_trajectories_with_rew_double_info(
    trajectories: Sequence[types.TrajectoryWithRew],
    limit_num_eps: Optional[int],
) -> DoubleInfoTransitionsWithRew:
    """Flatten trajectories into transitions with info dicts for obs and next_obs.

    Note: this will not include the final transition of the trajectory, because the
    info dictionary grouped with the final observation actually contains state
    information pertaining to the next episode.
    """
    keys = ["obs", "next_obs", "acts", "rews", "dones", "infos", "next_infos"]
    parts = {key: [] for key in keys}
    long_trajs = filter(lambda traj: len(traj.acts) > 2, trajectories)
    if limit_num_eps is not None:
        long_trajs = long_trajs[0:limit_num_eps]
    for traj in long_trajs:
        parts["acts"].append(traj.acts[1:-1])
        parts["obs"].append(traj.obs[1:-2])
        parts["next_obs"].append(traj.obs[2:-1])
        dones = np.zeros(len(traj.acts) - 2, dtype=bool)
        parts["dones"].append(dones)
        parts["rews"].append(traj.rews[1:-1])

        if traj.infos is None:
            infos = np.array([{}] * (len(traj) - 1))
            next_infos = np.array([{}] * (len(traj) - 1))
        else:
            infos = traj.infos[:-2]
            next_infos = traj.infos[1:-1]

        parts["infos"].append(infos)
        parts["next_infos"].append(next_infos)

    cat_parts = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return DoubleInfoTransitionsWithRew(**cat_parts)


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, Union[np.ndarray, th.Tensor]]:
    """Custom `th.utils.data.DataLoader` collate_fn for `DoubleInfoTransitionsWithRew`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument. Modified from imitation.data.types.

    Args:
        batch: The batch to collate.

    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    batch_no_infos = [
        {k: np.array(v) for k, v in sample.items() if k not in ["infos", "next_infos"]}
        for sample in batch
    ]

    result = torch_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    result["next_infos"] = [sample["next_infos"] for sample in batch]
    return result


def copy_module(mod: nn.Module) -> nn.Module:
    """Return a new module of the same type as mod, but with reset parameters."""
    copy_mod = copy.deepcopy(mod)
    if hasattr(copy_mod, "reset_parameters"):
        copy_mod.reset_parameters()
    for descendent_mod in copy_mod.modules():
        if hasattr(descendent_mod, "reset_parameters"):
            descendent_mod.reset_parameters()
    return copy_mod

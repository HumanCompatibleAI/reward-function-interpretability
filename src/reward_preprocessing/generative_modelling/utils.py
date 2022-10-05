# Utilities for training a generative model on imitation rollouts.
# Ideally, you should only need to export rollouts_to_dataloader.

from imitation.data import rollout, types
import numpy as np
from torch.utils import data as torch_data


def make_transition_to_tensor(num_acts):
    """Produces a function that takes a transition, produces a tensor.

    For use as something to 'map over' a torch dataset of transitions. Assumes
    observations are (h,c,w)-formatted images, actions are discrete.

    Args:
        num_acts: Number of discrete actions. Necessary because actions are
            saved as integers rather than one-hot vectors.
    """

    def transition_to_tensor(transition):
        obs = transition["obs"]
        act = transition["acts"]
        next_obs = transition["next_obs"]
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


def rollouts_to_dataloader(rollouts_paths, num_acts, batch_size):
    """Take saved rollouts of a policy, and produce a dataloader of transitions.

    Assumes that observations are (h,w,c)-formatted images and that actions are
    discrete.

    Args:
        rollouts_path: Path to rollouts saved via imitation script, or list of
            such paths.
        num_acts: Number of actions available to the agent (necessary because
            actions are saved as a number, not as a one-hot vector).
        batch_size: Int, size of batches that the dataloader serves. Note that
            a batch size of 2 will make the GAN algorithm think each batch is
            a (data, label) pair, which will mess up training.
    """
    if isinstance(rollouts_paths, list):
        rollouts = []
        for path in rollouts_paths:
            rollouts += types.load_with_rewards(path)
    else:
        rollouts = types.load_with_rewards(rollouts_paths)
    flat_rollouts = rollout.flatten_trajectories_with_rew(rollouts)
    tensor_rollouts = TransformedDataset(
        flat_rollouts, make_transition_to_tensor(num_acts)
    )
    rollout_dataloader = torch_data.DataLoader(
        tensor_rollouts, shuffle=True, batch_size=batch_size
    )
    return rollout_dataloader

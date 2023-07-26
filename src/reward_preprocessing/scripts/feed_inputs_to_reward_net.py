# Script to feed hand-crafted png files as inputs into reward networks.

import os
import os.path as osp

import PIL.Image as Image
import numpy as np
import torch as th
import torch.nn.functional as func


def feed_inputs_one_obs(reward_net_path, super_dir, num_acts):
    """
    Takes a reward net and a folder whose subfolders contain png images,
    then feeds the images to the reward net and prints the reward.

    Note: this only works if the reward net doesn't use the next_obs
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    reward_net = th.load(reward_net_path, map_location=device)
    reward_net.eval()
    for sub_dir in os.listdir(super_dir):
        print("\nevaluating images of type: ", sub_dir)
        rewards = []
        for image_name in os.listdir(osp.join(super_dir, sub_dir)):
            if image_name.endswith(".png"):
                image_path = osp.join(super_dir, sub_dir, image_name)
                image = np.asarray(Image.open(image_path))
                image = image[np.newaxis, :, :, 0:3]
                assert image.shape == (
                    1,
                    64,
                    64,
                    3,
                ), f"image shape is wrong: {image.shape}"
                image = np.concatenate([image for _ in range(num_acts)], axis=0)
                image = th.from_numpy(image).float().to(device) / 255.0
                action_nums = th.tensor(list(range(num_acts))).to(device)
                actions = func.one_hot(action_nums, num_classes=num_acts)
                reward = reward_net(image, actions, image, done=False)
                if num_acts == 1:
                    rewards.append(reward.item())
                else:
                    print(reward)
            else:
                continue
        if num_acts == 1:
            rewards = np.array(rewards)
            print("mean reward: ", np.mean(rewards))
            print("reward std: ", np.std(rewards))


def feed_inputs_two_obs(reward_net_path, super_dir, num_acts):
    """
    Takes a reward net and a folder whose sub-sub-folders contain pairs of
    images, named obs.png and next_obs.png, then feeds those images to the
    reward net and prints the reward.

    Note: this only works if the reward net uses both obs and next_obs
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    reward_net = th.load(reward_net_path, map_location=device)
    reward_net.eval()
    for sub_dir in os.listdir(super_dir):
        print("\nevaluating images of type: ", sub_dir)
        rewards = []
        for sub_sub_dir in os.listdir(osp.join(super_dir, sub_dir)):
            obs_in_dir = "obs.png" in os.listdir(
                osp.join(super_dir, sub_dir, sub_sub_dir)
            )
            assert obs_in_dir, "obs image not found"
            next_obs_in_dir = "next_obs.png" in os.listdir(
                osp.join(super_dir, sub_dir, sub_sub_dir)
            )
            assert next_obs_in_dir, "next_obs image not found"
            obs_path = osp.join(super_dir, sub_dir, sub_sub_dir, "obs.png")
            next_obs_path = osp.join(super_dir, sub_dir, sub_sub_dir, "next_obs.png")
            obs = np.asarray(Image.open(obs_path))
            next_obs = np.asarray(Image.open(next_obs_path))
            obs = obs[np.newaxis, :, :, 0:3]
            next_obs = next_obs[np.newaxis, :, :, 0:3]
            assert obs.shape == (1, 64, 64, 3), f"obs shape is wrong: {obs.shape}"
            next_obs_shape = next_obs.shape == (1, 64, 64, 3)
            assert next_obs_shape, f"next_obs shape is wrong: {next_obs.shape}"
            obs = np.concatenate([obs for _ in range(num_acts)], axis=0)
            next_obs = np.concatenate([next_obs for _ in range(num_acts)], axis=0)
            obs = th.from_numpy(obs).float().to(device) / 255.0
            next_obs = th.from_numpy(next_obs).float().to(device) / 255.0
            action_nums = th.tensor(list(range(num_acts))).to(device)
            actions = func.one_hot(action_nums, num_classes=num_acts)
            reward = reward_net(obs, actions, next_obs, done=False)
            if num_acts == 1:
                rewards.append(reward.item())
            else:
                rewards += reward.tolist()
        rewards = np.array(rewards)
        print("mean reward: ", np.mean(rewards))
        print("reward std: ", np.std(rewards))


if __name__ == "__main__":
    small_state_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/"
        + "procgen:procgen-coinrun-v0/20221130_100346_041de7/"
        + "checkpoints/00183/model.pt"
    )
    small_state_action_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/"
        + "procgen:procgen-coinrun-v0/20221130_112434_bdf145/"
        + "checkpoints/00174/model.pt"
    )
    large_state_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/"
        + "procgen:procgen-coinrun-v0/20221205_195843_b8bc1f/"
        + "checkpoints/00026/model.pt"
    )
    large_state_next_state_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/"
        + "procgen:procgen-coinrun-v0/"
        + "20221210_205125_06159b/"
        + "checkpoints/00004/model.pt"
    )
    large_all_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/"
        + "procgen:procgen-coinrun-v0/20221130_121635_89ed71/"
        + "checkpoints/00015/model.pt"
    )
    adversarially_trained_path = (
        "/nas/ucb/daniel/nas_reward_function_interpretability/output/train_regression/"
        + "procgen:procgen-coinrun-final-obs-v0/20230530_011818_8f2886/checkpoints/"
        + "final/model.pt"
    )
    reward_net_path = adversarially_trained_path
    num_acts = 15
    one_obs = False
    if one_obs:
        super_dir = "/nas/ucb/daniel/procgen_photoshops/single_obs"
        feed_inputs_one_obs(reward_net_path, super_dir, num_acts)
    else:
        super_dir = "/nas/ucb/daniel/procgen_photoshops/double_obs"
        feed_inputs_two_obs(reward_net_path, super_dir, num_acts)

"""Experiment that creates a dataset of transition-reward pairs,
which can then be used to train a reward model in a supervised fashion.
"""
from pathlib import Path

import numpy as np
from sacred import Experiment
from stable_baselines3 import PPO

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.transition import get_transitions

ex = Experiment("create_rollouts", ingredients=[env_ingredient])


@ex.config
def config():
    # path where the agent is saved (without .zip extension)
    model_path = ""
    # path where the dataset should be saved to (without .npz extension)
    save_path = ""
    train_samples = 10000
    test_samples = 10000
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(model_path: str, save_path: str, train_samples: int, test_samples: int):
    env = create_env()
    model = PPO.load(model_path)

    states = {}
    actions = {}
    next_states = {}
    rewards = {}
    dones = {}

    for mode, num_samples in zip(["train", "test"], [train_samples, test_samples]):
        states[mode] = []
        actions[mode] = []
        next_states[mode] = []
        rewards[mode] = []
        dones[mode] = []
        for transition, reward in get_transitions(env, model, num=num_samples):
            states[mode].append(transition.state)
            actions[mode].append(transition.action)
            next_states[mode].append(transition.next_state)
            rewards[mode].append(reward)
            dones[mode].append(transition.done)

    env.close()

    path = Path(save_path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(path),
        train_states=np.stack(states["train"], axis=0),
        train_actions=np.array(actions["train"]),
        train_next_states=np.stack(next_states["train"], axis=0),
        train_rewards=np.array(rewards["train"]),
        train_dones=np.array(dones["train"]),
        test_states=np.stack(states["test"], axis=0),
        test_actions=np.array(actions["test"]),
        test_next_states=np.stack(next_states["test"], axis=0),
        test_rewards=np.array(rewards["test"]),
        test_dones=np.array(dones["test"]),
    )

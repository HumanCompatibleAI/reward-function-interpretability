from typing import Any, Mapping

from sacred import Experiment
import torch
import wandb

from reward_preprocessing.env import create_env, env_ingredient
from reward_preprocessing.interp import (
    add_fixed_potential,
    add_noise_potential,
    fixed_ingredient,
    noise_ingredient,
    plot_rewards,
    reward_ingredient,
    rollout_ingredient,
    sparsify,
    sparsify_ingredient,
    transition_ingredient,
    visualize_rollout,
    visualize_transitions,
)
from reward_preprocessing.utils import add_observers

ex = Experiment(
    "interpret",
    ingredients=[
        env_ingredient,
        rollout_ingredient,
        sparsify_ingredient,
        noise_ingredient,
        fixed_ingredient,
        transition_ingredient,
        reward_ingredient,
    ],
)
add_observers(ex)


@ex.config
def config():
    run_dir = "runs/interpret"
    model_path = None  # path to the model to be interpreted (with extension)
    gamma = 0.99  # discount rate (used for all potential shapings)
    model_type = "ss"  # type of reward model, either 's', 'sa', 'ss' or 'sas'
    model_kwargs = {}  # additional kwargs for BasicRewardNet
    wb = {}  # kwargs for wandb.init()

    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(
    model_path: str,
    gamma: float,
    model_type: str,
    wb: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    _config,
):
    env = create_env()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(model_path, map_location=device)

    use_wandb = False
    if wb:
        wandb.init(
            project="reward_preprocessing",
            job_type="interpret",
            config=_config,
            **wb,
        )
        use_wandb = True

    model = add_fixed_potential(model, gamma)
    model = add_noise_potential(model, gamma)
    model = sparsify(model, device=device, gamma=gamma, use_wandb=use_wandb)
    model.eval()
    plot_rewards(model)
    visualize_transitions(model)
    visualize_rollout(model)
    env.close()

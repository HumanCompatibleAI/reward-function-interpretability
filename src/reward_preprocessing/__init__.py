"""Register custom environments"""

import gym
import procgen  # noqa: F401

from reward_preprocessing.dummy_dots_distances_env import (  # noqa: F401, I001
    DotsAndDistsEnv,
)
import reward_preprocessing.procgen as rmi_procgen  # noqa: I001

# Procgen

# note that procgen was imported to add procgen environments to the gym registry

GYM_PROCGEN_ENV_SPECS = list(
    filter(rmi_procgen.supported_procgen_env, gym.envs.registry.all())
)
rmi_procgen.register_procgen_envs(GYM_PROCGEN_ENV_SPECS)

# Dots and Distances dummy environment

DOTS_DISTS_ENV_SIZES = [200]
for size in DOTS_DISTS_ENV_SIZES:
    gym.envs.register(
        id=f"DotsAndDists-{size}-v0",
        entry_point="reward_preprocessing.dummy_dots_distances_env:DotsAndDistsEnv",
        max_episode_steps=10_000,
        kwargs={"size": size},
    )

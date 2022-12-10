"""Register custom environments"""

import gym
import procgen  # noqa: F401

import reward_preprocessing.procgen as rmi_procgen  # noqa: I001

# Procgen

# note that procgen was imported to add procgen environments to the gym registry

GYM_PROCGEN_ENV_SPECS = list(
    filter(rmi_procgen.supported_procgen_env, gym.envs.registry.all())
)
rmi_procgen.register_procgen_envs(GYM_PROCGEN_ENV_SPECS)

"""Register custom environments"""

import gym

from reward_preprocessing import procgen

# Procgen

GYM_PROCGEN_ENV_SPECS = list(
    filter(procgen.supported_procgen_env, gym.envs.registry.all())
)
procgen.register_procgen_envs(GYM_PROCGEN_ENV_SPECS)

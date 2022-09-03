from stable_baselines3.common.policies import ActorCriticCnnPolicy

from reward_preprocessing.ext.impala import ImpalaModel


class ImpalaPolicy(ActorCriticCnnPolicy):
    """Impala CNN policy as used in procgen. Compatible with stable-baselines3 and
    imitation."""

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, features_extractor_class=ImpalaModel)

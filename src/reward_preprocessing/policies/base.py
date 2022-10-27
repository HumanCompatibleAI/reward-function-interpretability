from stable_baselines3.common.policies import ActorCriticCnnPolicy

from reward_preprocessing.ext.impala import ImpalaModel


class ImpalaPolicy(ActorCriticCnnPolicy):
    """Impala CNN policy as used in procgen. Compatible with stable-baselines3 and
    imitation."""

    def __init__(self, *args, **kwargs):
        """Builds ImpalaPolicy; arguments passed to `ActorCriticPolicy`."""
        # We override any provided `features_extractor_class` with our ImpalaModel.
        # TODO: Potentially we want to allow setting feature extractors without having
        # them be overwritten here.
        kwargs.update(dict(features_extractor_class=ImpalaModel))
        super().__init__(*args, **kwargs)

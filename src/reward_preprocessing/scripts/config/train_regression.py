from imitation.scripts.common import common, demonstrations
import sacred

from reward_preprocessing.scripts.common import supervised

train_regression_ex = sacred.Experiment(
    "train_regression",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        supervised.supervised_ingredient,
    ],
)


@train_regression_ex.config
def defaults():
    # Every checkpoint_epoch_interval epochs, save the model. Epochs start at 1.
    checkpoint_epoch_interval = 1
    locals()  # make flake8 happy

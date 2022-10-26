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
    # Apparently in sacred I need default values, if I want to be able to override them
    supervised = dict(net_kwargs=dict(hid_channels=(32, 64)))
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_next_state():
    supervised = dict(
        net_kwargs=dict(use_state=False, use_action=False, use_next_state=True)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_all():
    supervised = dict(
        net_kwargs=dict(use_state=True, use_action=True, use_next_state=True)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def small():
    supervised = dict(net_kwargs=dict(hid_channels=(32, 64)))
    locals()  # make flake8 happy

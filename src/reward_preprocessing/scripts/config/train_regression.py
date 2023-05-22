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


@train_regression_ex.named_config
def use_state():
    supervised = dict(
        net_kwargs=dict(use_state=True, use_action=False, use_next_state=False)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_state_action():
    supervised = dict(
        net_kwargs=dict(use_state=True, use_action=True, use_next_state=False)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_next_state():
    supervised = dict(
        net_kwargs=dict(use_state=False, use_action=False, use_next_state=True)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_state_next_state():
    supervised = dict(
        net_kwargs=dict(use_state=True, use_action=False, use_next_state=True)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def use_all():
    supervised = dict(
        net_kwargs=dict(use_state=True, use_action=True, use_next_state=True)
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def large_net():
    # Similar to AlexNet architecture, only in the number of convolutional layers and
    # the number of channels in them.
    supervised = dict(net_kwargs=dict(hid_channels=(96, 256, 384, 384, 256)))
    locals()  # make flake8 happy


@train_regression_ex.named_config
def very_large_net():
    # Net that has the same convolutional networks as the Impala net used to train
    # policies. This does not have the other bells and whistles of Impala, such as
    # residual connections.
    # This network is probably too unnecessarily large for predicting the rewards using
    # supervised learning.
    supervised = dict(
        net_kwargs=dict(
            hid_channels=(16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32)
        )
    )
    locals()  # make flake8 happy


@train_regression_ex.named_config
def adversarial_training():
    supervised = dict(
        adversarial=True,
        nonsense_reward=0.0,
        vis_frac_per_epoch=0.04,
        gradient_clip_percentile=0.99,  # TODO figure out a reasonable number for this
    )
    locals()

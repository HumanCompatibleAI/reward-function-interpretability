from imitation.scripts.common import common
import sacred

from reward_preprocessing.scripts.common import supervised

train_probe_ex = sacred.Experiment(
    "train_probe",
    ingredients=[
        common.common_ingredient,
        supervised.supervised_ingredient,
    ],
)


@train_probe_ex.config
def default():
    traj_path = None  # set this and following 4 variables in named configs or via CLI
    reward_net_path = None
    layer_name = None  # should be act0, act1, act2, act3, or act4
    num_probe_layers = 0
    attributes = None
    attr_dim = None
    attr_cap = None
    batch_size = 128
    num_epochs = 5
    compare_to_mean = True
    compare_to_random_net = True
    locals()  # make flake8 happy


@train_probe_ex.named_config
def coinrun_large_all():
    # doesn't set layer_name
    traj_path = "coinrun_rollouts_3e3_episodes_2023-04.npz"
    reward_net_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/procgen:procgen-coinrun-v0/"
        + "20221130_121635_89ed71/checkpoints/00015/model.pt"
    )
    attributes = ["agent_goal_vec_x", "agent_goal_vec_y"]
    attr_dim = 2
    # 'supervised' determines the architecture of the randomly initialized network - set
    # it to be the same as the architecture of the reward net being loaded.
    supervised = dict(
        net_kwargs=dict(
            use_state=True,
            use_action=True,
            use_next_state=True,
            hid_channels=(96, 256, 384, 384, 256),
        )
    )
    common = dict(env_name="procgen:procgen-coinrun-final-obs-v0")
    locals()

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
    attr_func = None
    batch_size = 128
    num_epochs = 5
    compare_to_mean = True
    compare_to_random_net = True
    filter_extreme_attrs = False
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


@train_probe_ex.named_config
def heist_large_all():
    # doesn't set layer_name
    traj_path = (
        "/home/daniel/reward-function-interpretability/"
        + "heist_rollouts_3e3_episodes_2023-05.npz"
    )
    reward_net_path = (
        "/home/daniel/reward-function-interpretability/output/train_regression/"
        + "procgen:procgen-heist-final-obs-v0/20230504_161528_b8e9ba/checkpoints/"
        + "01119/model.pt"
    )
    attributes = "agent_goal_x"
    attr_dim = 1
    attr_cap = 13
    supervised = dict(
        net_kwargs=dict(
            use_state=True,
            use_action=True,
            use_next_state=True,
            hid_channels=(96, 256, 384, 384, 256),
        )
    )
    common = dict(env_name="procgen:procgen-heist-final-obs-v0")
    locals()


@train_probe_ex.named_config
def dots_and_dists():
    common = dict(env_name="DotsAndDists-64-v0")
    supervised = dict(
        net_kwargs=dict(
            use_state=True,
            use_action=True,
            use_next_state=True,
            hid_channels=(96, 256, 384, 384, 256),
        )
    )
    attributes = "distances"
    attr_dim = 3
    locals()


@train_probe_ex.named_config
def get_red():
    attr_func = lambda vec: vec[0]  # noqa: E731
    attr_dim = 1
    locals()


@train_probe_ex.named_config
def sort_distances():
    attr_func = sorted  # noqa: E731
    locals()


@train_probe_ex.named_config
def exp_distances():
    attr_func = lambda vec: [20**x for x in vec]  # noqa: E731
    locals()


@train_probe_ex.named_config
def sum_distances():
    attr_func = sum
    attr_dim = 1
    locals()

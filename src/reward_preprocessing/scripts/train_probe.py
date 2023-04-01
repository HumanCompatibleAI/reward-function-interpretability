from imitation.data import types
from imitation.scripts.common import demonstrations
import torch as th

from reward_preprocessing.common.utils import flatten_trajectories_with_rew_double_info
from reward_preprocessing.probes import Probe


def train_coinrun_probe(traj_path, reward_net_path):
    """Train a probe on the inverse coinrun agent-to-goal vector"""
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    reward_net = th.load(reward_net_path, map_location=device)

    probe = Probe(
        reward_net,
        layer_name="act4",
        attribute_dim=2,
        attribute_name=["agent_coin_vec_x", "agent_coin_vec_y"],
        loss_type="mse",
        device=device,
    )

    trajs = demonstrations.load_expert_trajs(
        rollout_path=traj_path,
        n_expert_demos=None,
    )
    assert isinstance(trajs[0], types.TrajectoryWithRew)
    dataset = flatten_trajectories_with_rew_double_info(trajs)

    probe.train(dataset=dataset, lr=0.01, frac_train=0.9, batch_size=64, num_epochs=10)


if __name__ == "__main__":
    traj_path = "coinrun_rollouts_5_episodes_2023-03.npz"
    net_path = (
        "/nas/ucb/pavel/out/interpret/train_regression/procgen:procgen-coinrun-v0/"
        + "20221130_121635_89ed71/checkpoints/00015/model.pt"
    )
    train_coinrun_probe(traj_path, net_path)

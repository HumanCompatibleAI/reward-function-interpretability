"""Quick ad-hoc script to train a RL agent using SB3."""
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

save_freq = 1_000_000
total_timesteps = 200_000_000

checkpoint_callback = CheckpointCallback(
    save_freq=save_freq, save_path="./out/sb3/checkpoints/"
)
eval_env = gym.make("procgen:procgen-coinrun-v0")
eval_callback = EvalCallback(
    eval_env,
    log_path="./out/sb3/results",
    eval_freq=save_freq,
)
# Create the callback list
callback = CallbackList([checkpoint_callback, eval_callback])


model = PPO("CnnPolicy", "procgen:procgen-coinrun-v0")
model.learn(total_timesteps=total_timesteps, callback=callback)
# Final save
model.save("ppo_coinrun_v0")

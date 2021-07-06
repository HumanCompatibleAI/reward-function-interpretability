from pathlib import Path
import tempfile

from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

from reward_preprocessing.env import create_env, env_ingredient

ex = Experiment("train_agent", ingredients=[env_ingredient])
ex.observers.append(FileStorageObserver("runs"))


@ex.config
def config():
    steps = 100000
    # If empty, the trained agent is only saved via Sacred observers
    # (you can still extract it manually later).
    # But if you already know you will need the trained model, then
    # set this to a filepath where you want the model to be stored,
    # without an extension (but including a filename).
    save_path = ""
    num_frames = 100
    _ = locals()  # make flake8 happy
    del _


@ex.automain
def main(steps: int, save_path: str, num_frames: int):
    env = create_env()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)

    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        # save the model
        if save_path == "":
            model_path = path / "trained_agent"
        else:
            model_path = Path(save_path)
        model.save(model_path)
        ex.add_artifact(model_path.with_suffix(".zip"))

        # record a video of the trained agent
        env = VecVideoRecorder(
            env,
            str(path),
            record_video_trigger=lambda x: x == 0,
            video_length=num_frames,
            name_prefix="trained_agent",
        )
        obs = env.reset()
        for _ in range(num_frames + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

        env.close()
        video_path = Path(env.video_recorder.path)
        ex.add_artifact(video_path)
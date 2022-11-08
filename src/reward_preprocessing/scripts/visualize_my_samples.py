import torch as th

from reward_preprocessing.common import utils

GAN_TIMESTAMP = "20221104_163134"
MODEL_NUMBER = "13720"

if __name__ == "__main__":
    gan_path = (
        "/nas/ucb/daniel/gan_test_data_"
        + GAN_TIMESTAMP
        + "/models/model_"
        + MODEL_NUMBER
        + ".torch"
    )
    device = "cuda" if th.cuda.is_available() else "cpu"
    gan = th.load(gan_path, map_location=th.device(device))
    samples, _ = gan.get_training_results()
    utils.visualize_samples(samples.detach().cpu().numpy(), gan.folder)

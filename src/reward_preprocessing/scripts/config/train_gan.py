"""Configuration settings for train_gan, training a generative model of transitions."""

import sacred
import vegans.GAN

from reward_preprocessing.generative_modelling import gen_models

train_gan_ex = sacred.Experiment("train_gan")


@train_gan_ex.config
def train_gan_defaults():
    generator_class = gen_models.Small21To84Generator
    discriminator_class = gen_models.SmallWassersteinCritic
    gan_algorithm = vegans.GAN.WassersteinGAN
    optim_kwargs = {
        "Generator": {"lr": 5e-4},
        "Adversary": {"lr": 1e-4},
    }  # keyword arguments for GAN optimizer
    num_training_epochs = 50
    batch_size = 256  # batch size for transition dataloader
    latent_shape = [3, 21, 21]  # shape of latent vector input to generator
    locals()  # make flake8 happy


@train_gan_ex.named_config
def procgen():
    generator_class = gen_models.FourTo64Generator
    latent_shape = [3, 4, 4]
    locals()

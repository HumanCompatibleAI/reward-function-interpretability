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
    generator_class = gen_models.DCGanFourTo64Generator
    discriminator_class = gen_models.DCGanWassersteinCritic
    gan_algorithm = vegans.GAN.WassersteinGANGP
    optim_kwargs = {
        "Generator": {"lr": 1e-4, "betas": (0.5, 0.9)},
        "Adversary": {"lr": 1e-4, "betas": (0.5, 0.9), "weight_decay": 1e-3},
    }
    num_training_epochs = 1
    batch_size = 128
    latent_shape = [100]
    print_every = "0.01e"
    save_losses_every = "0.1e"
    save_model_every = "0.1e"
    num_acts = 15
    device = "cuda"
    locals()

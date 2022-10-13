import torch as th
import torch.nn as nn

# TODO: add test on initialization that shit has the right shape?


class Small21To84Generator(nn.Module):
    """
    Small generative model that takes 21 x 21 noise to an 84 x 84 image.
    """

    def __init__(self, latent_shape, data_shape):
        super(Small21To84Generator, self).__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(latent_shape[0], 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
        )
        self.output = nn.Conv2d(32, data_shape[0], kernel_size=3, padding=1)

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.output(x)
        return x


class SmallFourTo64Generator(nn.Module):
    """
    Small generative model that takes 4 x 4 noise to a 64 x 64 image.

    Of use for generative modelling of procgen rollouts.
    """

    def __init__(self, latent_shape, data_shape):
        super(SmallFourTo64Generator, self).__init__()
        self.hidden_part = nn.Sequential(
            nn.ConvTranspose2d(latent_shape[0], 32, kernel_size=4, padding=1, stride=2),
            # now 8x8
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            # now 16x16
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            # now 32x32
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            # now 64x64
            nn.LeakyReLU(0.1),
        )
        self.output = nn.Conv2d(32, data_shape[0], kernel_size=3, padding=1)

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.output(x)
        return x


class DCGanFourTo64Generator(nn.Module):
    """
    DCGAN-based generative model that takes a 1-D latent vector to a 64x64 image.

    Of use for generative modelling of procgen rollouts.
    """

    def __init__(self, latent_shape, data_shape):
        super(DCGanFourTo64Generator, self).__init__()
        self.project = nn.Linear(latent_shape[0], 1024 * 4 * 4)
        self.conv_body = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1, stride=2),
            # now 8x8
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2),
            # now 16x16
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2),
            # now 32x32
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, data_shape[0], kernel_size=4, padding=1, stride=2),
            # now 64x64
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.project(x)
        x = th.reshape(x, (batch_size, 1024, 4, 4))
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv_body(x)
        return x


class SmallWassersteinCritic(nn.Module):
    """
    Small critic for use in the Wasserstein GAN algorithm.
    """

    def __init__(self, data_shape):
        super(SmallWassersteinCritic, self).__init__()
        self.hidden_part = nn.Sequential(
            nn.Conv2d(data_shape[0], 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )
        self.output = nn.Identity()

    def forward(self, x):
        x = self.hidden_part(x)
        x = self.output(x)
        return x


class DCGanWassersteinCritic(nn.Module):
    """
    Wasserstein-GAN critic based off the DCGAN architecture.
    """

    def __init__(self, data_shape):
        super(DCGanWassersteinCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(data_shape[0], 128, kernel_size=4, padding=1, stride=2),
            # now 32 x 32
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            # now 16 x 16
            nn.LeakyReLU(0.1),
            nn.LayerNorm([256, 16, 16]),
            nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2),
            # now 8 x 8
            nn.LeakyReLU(0.1),
            nn.LayerNorm([512, 8, 8]),
            nn.Conv2d(512, 1024, kernel_size=4, padding=1, stride=2),
            # now 4 x 4
            nn.LeakyReLU(0.1),
            nn.LayerNorm([1024, 4, 4]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.network(x)

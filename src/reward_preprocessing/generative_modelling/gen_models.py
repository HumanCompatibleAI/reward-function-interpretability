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

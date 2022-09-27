"""Train a generative model of transitions.

For use in reward function feature visualization.
"""

from reward_preprocessing.generative_modelling import utils
from reward_preprocessing.scripts.config.train_gan import train_gan_ex

# rollouts_path, num_acts not specified


@train_gan_ex.main
def train_gan(
    generator_class,
    discriminator_class,
    gan_algorithm,
    optim_kwargs,
    rollouts_path,
    num_acts,
    num_training_epochs,
    batch_size,
    latent_shape,
    gan_save_path,
    device="cpu",
    ngpu=None,
    steps={"Adversary": 5},
):
    """TODO document these

    latent_shape has to be (c,h,w)
    gan_save_path is a folder where a bunch of stuff relevant to GAN training
        will be dumped. Should be empty.
    device: "cpu" or "cuda"
    ngpu: number of GPUs

    TODO decide whether to have GAN verbosity be an arg - probably make it a
    default kwarg? (by which I mean how many times to print loss etc, how
    many times to save loss)
    """
    transitions_loader = utils.rollouts_to_dataloader(
        rollouts_path, num_acts, batch_size
    )
    transitions_batch = next(iter(transitions_loader))
    trans_shape = list(transitions_batch.shape)[1:]
    generator = generator_class(latent_shape, trans_shape)
    discriminator = discriminator_class(trans_shape)
    gan = gan_algorithm(
        generator,
        discriminator,
        z_dim=latent_shape,
        x_dim=trans_shape,
        optim_kwargs=optim_kwargs,
        folder=gan_save_path,
        device=device,
        ngpu=ngpu,
    )
    gan.fit(
        transitions_loader,
        batch_size=batch_size,
        print_every="2e",
        save_losses_every="0.25e",
        epochs=num_training_epochs,
        steps=steps,
    )


# TODO: write script to use this in feature viz.

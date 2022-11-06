"""Train a generative model of transitions.

For use in reward function feature visualization.
"""

from sacred.observers import FileStorageObserver
import torch as th

from reward_preprocessing.common import utils
from reward_preprocessing.scripts.config.train_gan import train_gan_ex

# TODO: write script to use this in feature viz.


@train_gan_ex.main
def train_gan(
    generator_class,
    discriminator_class,
    gan_algorithm,
    optim_kwargs,
    rollouts_paths,
    num_acts,
    num_training_epochs,
    batch_size,
    latent_shape,
    gan_save_path,
    device="cpu",
    ngpu=None,
    optimizer=th.optim.Adam,
    adv_steps=5,
    print_every="1e",
    save_losses_every="0.25e",
    save_model_every="1e",
):
    """Train a GAN on a set of transitions.

    Assumes that observations are image-shaped and actions are discrete.

    Args:
        generator_class: Upon initialization, takes a shape for the latent
            space and the shape of the transition tensors. Instantiates a
            network that takes latent vectors and returns transition tensors.
        discriminator_class: Upon initialization, takes a shape for the
            transition tensors. Instantiates a network that takes a
            transition tensor and gives it a realism score.
        gan_algorithm: A GAN training algorithm imported from `vegans`.
        optim_kwargs: A dictionary of keyword arguments for the generator and
            adversary networks.
        rollouts_paths: Path of rollouts saved by `imitation`, or list of paths.
        num_acts: Number of actions in the training environment.
        num_training_epochs: How many epochs to train the GAN for.
        batch_size: Number of transitions per batch to be trained on.
        latent_shape: Shape of the latent tensor to be fed into the generator
            network. Should be in (c,h,w) format.
        gan_save_path: Directory in which to save GAN training details.
        device: "cpu" or "cuda", depending on what you're training on.
        ngpu: Number of GPUs to train on, if training on GPUs.
        optimizer: torch.optim. Optimizer to train GAN with.
        adv_steps: Number of steps to train the adversary for for each step the
            generator is trained for.
        print_every: String specifying how many epochs should elapse between
            successive printings of training information.
        save_losses_every: String specifying how many epochs should elapse
            between successive savings of loss information.
        save_model_every: String specifying how many epochs should elapse
            between successive savings of the model.
    """
    # create data loader of transitions
    transitions_loader = utils.rollouts_to_dataloader(
        rollouts_paths, num_acts, batch_size
    )
    # define gan
    transitions_batch = next(iter(transitions_loader))
    trans_shape = list(transitions_batch.shape)[1:]
    generator = generator_class(latent_shape, trans_shape)
    discriminator = discriminator_class(trans_shape)
    gan = gan_algorithm(
        generator,
        discriminator,
        z_dim=latent_shape,
        x_dim=trans_shape,
        optim=optimizer,
        optim_kwargs=optim_kwargs,
        folder=gan_save_path,
        device=device,
        ngpu=ngpu,
    )
    # print out summary
    gan.summary()
    # fit gan
    steps = {"Adversary": adv_steps}
    gan.fit(
        transitions_loader,
        batch_size=batch_size,
        print_every=print_every,
        save_losses_every=save_losses_every,
        save_model_every=save_model_every,
        epochs=num_training_epochs,
        steps=steps,
    )
    # save samples, return losses, save plot of losses
    samples, losses = gan.get_training_results()
    utils.save_loss_plots(losses, gan.folder)
    utils.visualize_samples(samples, gan.folder)
    return losses


def main_console():
    observer = FileStorageObserver("train_gan")
    train_gan_ex.observers.append(observer)
    train_gan_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()

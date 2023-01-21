from imitation.scripts.common import common
from sacred import Experiment

interpret_ex = Experiment("interpret", ingredients=[common.common_ingredient])


@interpret_ex.config
def defaults():
    # Path to the learned supervised reward net
    reward_path = None
    # Rollouts to use for dataset visualization
    rollout_path = None
    # Limit the number of observations to use for dim reduction.
    # The RL Vision paper uses "a few thousand" observations.
    limit_num_obs = 2048
    # Whether to plot visualizations as pyplot figures. Set to False when running
    # interpret in a non-GUI environment, such as the cluster. In that case, use
    # wandb logging or save images to disk.
    pyplot = False
    vis_scale = 4  # Scale the visualization img by this factor
    # Type of visualization to do. Options are "traditional" (gradient-based),
    # "dataset", and "dataset_traditional" (first do dataset, then traditional).
    vis_type = "traditional"
    # Name of the layer to visualize. To figure this out run interpret and the
    # available layers will be printed. For additional notes see interpret doc comment.
    layer_name = "reshaped_out"
    # Number of features to use for dim reduction. No dim recution if None.
    num_features = None
    # Path to the GAN model. If None simply visualize reward net without the use of GAN.
    gan_path = None
    # Regularization of L2 norm of GAN latent vector. None if not using GAN. If using
    # GAN, best practice is to set this to ~1e-4
    l2_coeff = None
    # Directory to save images to. If specified, should end in a /.
    img_save_path = None
    # What regularization to use for generated images.
    reg = {
        "no_gan": {
            "jitter": 8,  # Jitter for generated images.
        }
    }

    locals()  # quieten flake8

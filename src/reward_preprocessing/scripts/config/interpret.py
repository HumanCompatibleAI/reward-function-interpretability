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
    pyplot = False  # Plot images as pyplot figures
    vis_scale = 4  # Scale the visualization img by this factor
    vis_type = "traditional"  # "traditional" (gradient-based) or "dataset"
    # Name of the layer to visualize. To figure this out run interpret and the
    # available layers will be printed. For additional notes see interpret doc comment.
    layer_name = "reshaped_out"
    # Number of features to use for dim reduction. No dim recution if None.
    num_features = None
    # Path to the GAN model. If None simply visualize reward net without the use of GAN.
    gan_path = None

    locals()  # quieten flake8

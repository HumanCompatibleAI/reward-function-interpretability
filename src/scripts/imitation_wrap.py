
import sys


# import magical
import sacred
from imitation.scripts import train_rl, eval_policy, train_adversarial, \
    train_imitation

# Register envs which are not used by default
# Not needed for procgen for some reason
# magical.register_envs()


def main():
    argv = sys.argv[1:]
    experiment_name = argv[0]

    experiment: sacred.Experiment
    if experiment_name == "train_rl":
        experiment = train_rl.train_rl_ex
    elif experiment_name == "eval_policy":
        experiment = eval_policy.eval_policy_ex
    elif experiment_name == "train_adversarial":
        experiment = train_adversarial.train_adversarial_ex
    elif experiment_name == "train_imitation":
        experiment = train_imitation.train_imitation_ex
    else:
        raise NotImplementedError(f"Experiment {experiment_name} not implemented!")

    experiment.run_commandline(argv)


if __name__ == "__main__":
    main()

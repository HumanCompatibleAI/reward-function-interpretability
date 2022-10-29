[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/reward-preprocessing/tree/main.svg?style=svg&circle-token=5689f087396d3f526afd49f3af9d4b098560f79c)](https://circleci.com/gh/HumanCompatibleAI/reward-preprocessing/tree/main)
# Reward Function Interpretability

## Installation
First clone this repository.
Install inside your python virtual environment system of choice:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If dev packages are needed.
```

Note on the python version: if you use a separate installation of `imitation`
for creating reward models, then make sure that your environment for `imitation`
and for `reward_preprocessing` use the same version of Python, otherwise
you might run into issues when unpickling `imitation` models for use in
`reward_preprocessing`.

## Docker

You can also use docker by building the image and running the scripts from inside the container.

## Usage

Use `print_config` to get details of the config params of the sacred scripts.

## Code Structure

- `src`
  - `notebooks`: Currently only a WIP port of the rl vision notebook.
  - `reward_preprocessing`
    - `common`: Code used across multiple submodules.
    - `env`: Implemented environments, mostly used for the old reward-preprocessing code.
    - `ext`: Code that is basically unmodified from other sources. Often this will be subject to a different license than the main repo.
      - However, this is not exclusive: There are other files which rely heavily on other sources and are therefore subject to a different license.
    - `policies`: RL policies for training experts with train_rl.
    - `preprocessing` Reward preprocessing / reward shaping code.
    - `scripts`: All scripts that are not the main scripts of the projects. Helpers and scripts that produce artifacts that are used by the main script. Everything here should either be an executable file or a config for one.
       - `helpers`: Helper scripts that are bash executables.
    - `trainers`: Our additions to the suite of reward learning algorithms available in imitation. Currently this contains the trainer for training reward nets with supervised learning.
    - `vis`: Visualization code for interpreting reward functions.
    - `interpret.py`: The main script that provides the functionality for this project.
- `tests`

## Philosophy on Dependencies

### Lucent

Unless there is an outright bug in `lucent`, when extending `lucent` add the code here and leave the dependency to vanilla lucent (see e.g. `reward_preprocessing.vis.objectives`).

### Imitation

So far we have tried to keep everything as compatible with `imitation` as possible to the point where this project could (almost) be an extension for imitation.
This also gives us imitation's logging and sacred integration for free.

## Tests
Run
```
poetry run pytest
```
to run all available tests (or just run `pytest` if the environment is active).

## Code style and linting
`ci/code_checks.sh` contains checks for formatting and linting.
We recommend using it as a pre-commit hook:
```
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```

To automatically change the formatting (rather than just checking it), run
```
ci/format_code.sh
```
Alternatively, you may want to configure your editor to run `black` and `isort` when saving a file.

Run `poetry run pytype` (or just `pytype` inside the `poetry` shell) to type check
the code base. This may take a bit longer than linting, which is why it's not part
of the pre-commit hook. It's still checked during CI though.

# Content From Old Reward Preprocessing Readme

## Running experiments
The experimental pipeline consists of three to four steps:
1. Create the reward models that should be visualized (either by using `imitation`
   to train models, or by using `create_models.py` to generate ground truth
   reward functions)
2. In a non-tabular setting: generate rollouts to use for preprocessing and
   visualization. This usually means training an expert policy and collecting rollouts
   from that.
3. Use `preprocess.py` to preprocess the reward function.
4. Use `plot_heatmaps.py` (for gridworlds) or `plot_reward_curves.py` (for other
   environments) to plot the preprocessed rewards.

Below, we describe these in more detail.

### Creating reward models
The input to our preprocessing pipeline is an `imitation` `RewardNet`,
saved as a `.pt` file. You can create one using any of the reward learning
algorithms in `imitation`, or your own implementation.

In our paper, we also have experiments on shaped versions of the ground truth
reward. To simplify the rest of the pipeline, we have implemented these
as simple static `RewardNet`s, which can be stored to disk using `create_models.py`.
(This will store all the different versions in `results/ground_truth_models`.)

### Creating rollouts
Tabular environments do not require samples for preprocessing, since we can optimize
the interpretability objective over the entire space of transitions. But for non-tabular
environments, a transition distribution to optimize over is needed. Specifically,
the preprocessing step requires a sequence of `imitation`s `TrajectoryWithRew`.

The source of these trajectories can be one of several options, as well as a combination
of those. But the most important cases are:
- a pickled file of trajectories, specified as `rollouts=[(0, None, "<rolout name>", "path/to/rollouts.pkl")]`
  in the command line arguments
- rollouts generated on the fly using some policy, specified as `rollouts=[(0, "path/to/policy.zip", "<rollout name>)]`

Since always generating rollouts on the fly is inefficient, we have the helper script `create_rollouts.py`
that takes some rollout configuration (such as the second one above) and produces a pickled file with
the corresponding rollouts, which can then be used during further stages using the first example above.

### Preprocessing
The main part of our pipeline is `preprocess.py`, which takes in a `rollouts=...` config as just described (not required
in a tabular setting) and a `model_path=path/to/reward_model.pt` config, and then produces preprocessed versions
of the input model (also stored as `.pt` files).

An example command for gridworld would be:
```python -m reward_preprocessing.preprocess with env.empty_maze_10 model_path=path/to/reward_net.pt save_path=path/to/output_folder```

The `env.empty_maze_10` part specifies the environment (in this case, using one of the preconfigured options).

## Directories
`runs/` and `results/` are both in `.gitignore` and are meant to be used to
store artifacts. `runs/` is used for the Sacred `FileObserver`, so it contains
information about every single run. In contrast, `results/` is meant for
generated artifacts that are meant to be used by subsequent runs, such as trained
models or datasets of transition-reward pairs. It will only contain the versions
of these artifacts that are needed (e.g. the best/latest model), whereas `runs/`
will contain all the artifacts generated by past runs.

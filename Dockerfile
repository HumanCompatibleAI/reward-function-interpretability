# base stage contains just dependencies.
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt list
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    libc6-dev \
    # Python
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    python3-virtualenv \
    # Used by procgen
    build-essential \
    libgl1-mesa-dev \
    qt5-default \
    cmake \
    # git is needed by Sacred
    git \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -V
ENV PATH="/venv/bin:$PATH"
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3.9 -


WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY poetry.lock pyproject.toml ./
# Ideally, we'd clear the poetry cache but this seems annoyingly
# difficult and not worth getting right for this project
# (I've tried some things from https://github.com/python-poetry/poetry/issues/521
# without success)
RUN poetry install --no-interaction --no-ansi
# HACK: make sure we have all the dependencies of imitation but then uninstall
# imitation itself. We want to use the git version of that and that changes too
# frequently to include in this stage
RUN poetry run pip install git+https://github.com/HumanCompatibleAI/imitation && poetry run pip uninstall --yes imitation && poetry run pip cache purge

# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm poetry.lock pyproject.toml

# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
RUN poetry run pip install git+https://github.com/HumanCompatibleAI/imitation
# This commit, which is newer than the newest pypi release, fixes a bug related to the number of args given to the seed function.
RUN poetry run pip install git+https://github.com/openai/gym3.git@4c3824680eaf9dd04dce224ee3d4856429878226
# Newer version with modification since the preprint was released.
RUN poetry run pip install git+https://github.com/JacobPfau/procgenAISC.git@f4b4be55337ad8e4e6d2f13337534f9911a40277

# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
RUN poetry run python3 setup.py sdist bdist_wheel
RUN poetry run pip install dist/reward_preprocessing-*.whl

CMD ["poetry", "run", "pytest"]

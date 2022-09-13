# base stage contains just dependencies.
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.9
    python3.9 \
    python3.9-dev \
    python3-pip \
    virtualenv \
    # Misc
    curl \
    gcc \
    g++ \
    libc6-dev \
    # git is needed by Sacred
    git \
    # Needed for procgen
    qt5-default \
    cmake \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/venv/bin:$PATH"
COPY ci/build_and_activate_venv.sh ./ci/build_and_activate_venv.sh
RUN ci/build_and_activate_venv.sh /venv

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"


WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY poetry.lock pyproject.toml ./
# Ideally, we'd clear the poetry cache but this seems annoyingly
# difficult and not worth getting right for this project
# (I've tried some things from https://github.com/python-poetry/poetry/issues/521
# without success)
#RUN pip install "gym3 @ git+https://github.com/openai/gym3.git@4c38246"
RUN poetry install --no-interaction --no-ansi
# We have to install procgen using poetry because it relies on gym3 in its build.py which is only installed inside the poetry env
RUN poetry run pip install "procgen @ git+https://github.com/JacobPfau/procgenAISC.git@822dd97"

# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm poetry.lock pyproject.toml

# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
RUN poetry run python setup.py sdist bdist_wheel
RUN poetry run pip install dist/reward_preprocessing-*.whl

CMD ["poetry", "run", "pytest"]

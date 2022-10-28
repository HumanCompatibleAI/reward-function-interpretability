# base stage contains just dependencies.
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.9
    python3.9 \
    python3.9-dev \
    python3-pip \
    python-tk \
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

WORKDIR /reward_preprocessing

ENV PATH="/venv/bin:$PATH"
COPY ./requirements.txt /reward_preprocessing/
COPY ./requirements-dev.txt /reward_preprocessing/
COPY ci/build_and_activate_venv.sh ./ci/build_and_activate_venv.sh
RUN ci/build_and_activate_venv.sh /venv


# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
RUN python setup.py sdist bdist_wheel
RUN pip install dist/reward_preprocessing-*.whl

CMD ["pytest"]

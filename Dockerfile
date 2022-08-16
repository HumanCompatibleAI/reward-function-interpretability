# base stage contains just dependencies.
FROM python:3.9.5-slim as dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    libc6-dev \
    # git is needed by Sacred
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
ENV PATH="/root/.local/bin:$PATH"


WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY poetry.lock pyproject.toml ./
# Ideally, we'd clear the poetry cache but this seems annoyingly
# difficult and not worth getting right for this project
# (I've tried some things from https://github.com/python-poetry/poetry/issues/521
# without success)
RUN poetry install --no-interaction --no-ansi

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

#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=$1
if [[ ${venv} == "" ]]; then
	venv="venv"
fi

virtualenv -p python3.9.5 ${venv}
source ${venv}/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

if [[ $USE_MPI == "True" ]]; then
  pip install mpi4py
fi
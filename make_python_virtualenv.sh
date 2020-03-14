#!/usr/bin/env bash

# Makes Python 3 virtual environment in a directory called .env
# Tested on OSX (needs python3, pip3, and virtualenv installed)

pip3 install -U pip3
pip3 install -U setuptools

pathToPython3=$(which python3)
echo $pathToPython3

python3 -m venv .env
#virtualenv -p $pathToPython3 .env
#venv -p $pathToPython3 .env

source .env/bin/activate

pip3 install -r requirements.txt


deactivate

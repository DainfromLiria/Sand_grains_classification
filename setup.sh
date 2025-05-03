#!/bin/bash

if command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is installed"
    python3 --version
else
    echo "Python 3 is not installed! Please install it."
fi


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-dependencies git+https://github.com/nasa/pretrained-microscopy-models

# shellcheck disable=SC2164
cd scripts
python3 merge_split_pt.py merge
echo "model.pt was successfully created"
# shellcheck disable=SC2103
cd ..

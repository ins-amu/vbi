#!/bin/bash
set -eux
echo "Setting up VBI environment for EBRAINS..."
rm -rf /tmp/vbi
python3 -m venv /tmp/vbi
unset PYTHONPATH
source /tmp/vbi/bin/activate
pip install ipykernel scikit_learn matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sbi --no-deps
pip install pyro-ppl tensorboard nflows pyknos zuko arviz pymc
mkdir -p /tmp/src && cd /tmp/src
rm -rf vbi
git clone --depth 1 https://github.com/ins-amu/vbi
cd vbi
pip install -e .
python -m ipykernel install --user --name VBI
echo "Setup complete! Please reload browser and select VBI kernel."

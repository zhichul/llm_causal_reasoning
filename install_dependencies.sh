#!/bin/bash
set -e

if [ ! -f "install_dependencies.sh" ]; then
    echo "Please run this script in the root folder of the repo."
    exit 1
fi

# params
verl_commit=15263cb86a464264edb1e5462675e25ddf6ff9d8
proj_root=$(pwd)

# push some environment variables
echo "ROOT=$(pwd)" > .env

# create conda env
conda create -n scgl python==3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scgl
conda env list

# get verl
if [ -e "lib/verl" ]; then
    echo "Verl already downloaded, skipping clone."
else
    git clone git@github.com:volcengine/verl.git lib/verl
fi
cd lib/verl
git checkout $verl_commit
pip3 install -e .
cd $proj_root
pip install vllm==0.6.3
# you may need this if flash-attn refuses to install
# conda install -c nvidia cuda-toolkit=12.1
# PATH=$CONDA_PREFIX/bin:$PATH
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install flash_attn --no-build-isolation

# After this you may see a pip complaint about `tensordict` version being too new
# I ignored it for now without running any issues so far

pip install -r requirements.txt

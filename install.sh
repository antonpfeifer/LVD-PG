#!/bin/bash

# install julia
if ! command -v julia &> /dev/null
then
    echo "Julia could not be found, installing Julia..."
    # Download and install Julia (you can change the version as needed)
    JULIA_VERSION="1.8.5"
    JULIA_INSTALL_DIR="$HOME/julia-$JULIA_VERSION"
    if [ ! -d "$JULIA_INSTALL_DIR" ]; then
        curl -fsSL https://install.julialang.org | sh
    fi
    export PATH="$JULIA_INSTALL_DIR/bin:$PATH"
else
    echo "Julia is already installed."
fi




# ensure that python 3 is used
python3 --version

# create virtual environment
conda create --file environment.yml

# activate virtual environment and install dependencies
conda activate lvd-pg

# install julia dependencies discovered from Julia source imports
julia --project=. -e 'using Pkg; Pkg.instantiate(); ENV["PYTHON"] = "/home/anton/miniconda3/envs/lvd-pg/bin/python"; Pkg.build("PyCall")'

mkdir -p exps/progressive_growing/data/data_imagenet32
touch exps/progressive_growing/data/data_imagenet32/data_trn.npy
touch exps/progressive_growing/data/data_imagenet32/data_val.npy
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --partition=gpuexpress
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

#SBATCH --job-name=lvd-pg
#SBATCH --output=output.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpfeife3@uni-muenster.de

# load modules with available GPU support (this is an example, modify to your needs!)
module load CUDA
module load uv

# run your application
./install.sh

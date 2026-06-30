#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

#SBATCH --job-name=vqvae_train
#SBATCH --output=vqvae_train_160626.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

srun python3 train_vqvae2_model.py -id --data-path "/scratch/tmp/mpfeife3/bachelorarbeit/data/data"

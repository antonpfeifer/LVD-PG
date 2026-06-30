#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

#SBATCH --job-name=imagenet_finetune
#SBATCH --output=imagenet_finetune_160626.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

srun bash finetune.sh

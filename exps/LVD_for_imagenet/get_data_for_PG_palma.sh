#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1
#SBATCH --mem=200G

#SBATCH --job-name=imagenet_augementation
#SBATCH --output=imagenet_augemntation_160626_3.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

export DATA_PATH=/scratch/tmp/mpfeife3/bachelorarbeit/data/data

set -e
source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

srun python3 get_data_for_PG.py -id --data-path "$DATA_PATH"

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1
#SBATCH --mem=300G

#SBATCH --job-name=wikitext_top_level_train
#SBATCH --output=../temp/logs/wikitext_top_level_300626.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

set -a
source ../../.env
set +a

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

srun python ../progressive_growing_top.py

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1
#SBATCH --mem=300G

#SBATCH --job-name=wikitext_bert_annotation
#SBATCH --output=get_data_for_PG_160626.dat
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
srun python ../get_data_for_PG_sentence.py --max-sentences 100000 --teacher-model Qwen/Qwen-3-VL-Embedding-2B --batch-size 16

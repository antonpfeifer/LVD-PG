#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --job-name=wikitext_bert_annotation
#SBATCH --output=get_data_for_PG_160626.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

set -a
source ../../.env
set +a

export SENTENCE_TRANSFORMERS_HOME="/scratch/tmp/mpfeife3/bachelorarbeit"
export HUGGINGFACE_HUB_CACHE="/scratch/tmp/mpfeife3/bachelorarbeit/hf_cache"

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2
srun python ../get_data_for_PG_sentence.py --max-sentences 100000 --teacher-model "Qwen/Qwen3-VL-Embedding-2B" --batch-size 16

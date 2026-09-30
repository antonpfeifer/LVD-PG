#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --mem=300G

#SBATCH --job-name=wikitext_bert_annotation
#SBATCH --output=get_data_for_PG_10M.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

set -a
source ../../.env
set +a

export BASE_DIR="/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg"

export SENTENCE_TRANSFORMERS_HOME="/scratch/tmp/mpfeife3/bachelorarbeit"
export HUGGINGFACE_HUB_CACHE="/scratch/tmp/mpfeife3/bachelorarbeit/hf_cache"
export TOKENIZER_DOWNLOAD_DIR="/scratch/tmp/mpfeife3/bachelorarbeit/tokenizer_cache"
# HF_TOKEN must be provided via ../../.env (gitignored); do not hardcode it here.

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

# Preflight: fail fast if torch cannot see the GPU instead of burning CPU hours
nvidia-smi || { echo "ERROR: nvidia-smi failed, no GPU visible"; exit 1; }
python -c "import torch, sys; print(f'torch={torch.__version__} cuda_build={torch.version.cuda} available={torch.cuda.is_available()}'); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: CPU-only torch (torch.cuda.is_available()=False). Fix env: conda-forge cpu_generic pytorch shadows pip cu126 build."; exit 1; }

srun python ../get_data_for_PG_sentence.py --max-sentences 10000000 --teacher-model "codefuse-ai/F2LLM-4B" --batch-size 8192 --output-dir "${BASE_DIR}/exps/LVD_for_wikitext/data/data_wikitext/10M"

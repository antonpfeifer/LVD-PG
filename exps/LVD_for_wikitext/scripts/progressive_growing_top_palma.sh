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

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

ml palma/2024a
ml GCC/13.3.0
ml CUDA/13.0.2

# Slurm may run a copied batch script from /var/spool, so do not use
# BASH_SOURCE[0] to locate the project files.  Use the submission directory.
if [[ -f "${SLURM_SUBMIT_DIR}/progressive_growing_top.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/scripts/progressive_growing_top.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
else
  echo "Could not find progressive_growing_top.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi

EXPS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
REPO_ROOT="$(realpath "${EXPS_ROOT}/..")"

srun python "${SCRIPT_DIR}/progressive_growing_top.py" \
  --julia-project "${REPO_ROOT}" \
  --data-root "${EXPS_ROOT}/progressive_growing/data" \
  --temp-root "${EXPS_ROOT}/progressive_growing/temp"

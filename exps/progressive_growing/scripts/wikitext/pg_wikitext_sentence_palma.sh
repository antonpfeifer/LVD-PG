#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

#SBATCH --job-name=wikitext_sentence
#SBATCH --output=/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing/temp/logs/pg_wikitext_sentence_02.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

#SBATCH --mem=300G
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:4

# NOTE: sbatch executes a *copy* of this script from its spool dir
# (/var/spool/slurm/...), so BASH_SOURCE[0] does NOT point at the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from; fall back to
# the script location for manual runs outside Slurm.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

srun bash "${SUBMIT_DIR}/pg_wikitext_sentence.sh"

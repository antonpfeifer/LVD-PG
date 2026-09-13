#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

#SBATCH --job-name=wikitext_sentence
#SBATCH --output=/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing/temp/logs/pg_wikitext_070726.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

#SBATCH --mem=256G
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1

BASE_PATH="/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing"

srun bash pg_wikitext_sentence.sh

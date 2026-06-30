#!/bin/sh

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

#SBATCH --job-name=wikitext_train_pc_more_cats
#SBATCH --output=../../temp/logs/pg_wikitext_out_230626_33278_cats.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

#SBATCH --mem=128G
#SBATCH --partition=gpuh200mini
#SBATCH --gres=gpu:1

srun bash pg_wikitext.sh

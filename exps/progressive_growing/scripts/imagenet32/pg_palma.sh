#!/bin/sh

#SBATCH --mem=200G
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=anton.pfeifer@uni-muenster.de

#SBATCH --output=../../temp/logs/pg_imagenet32.dat
#SBATCH --job-name=pg_imagenet32

bash pg.sh "imagenet32"

#!/bin/sh

#SBATCH --mem=200G
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:2

bash pg.sh "imagenet32"

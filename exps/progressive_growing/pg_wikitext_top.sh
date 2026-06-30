#!/bin/usr/env bash
set -e
source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

export PYTHON=/scratch/tmp/mpfeife3/bachelorarbeit/miniconda/envs/lvd-pg/bin/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/julia:$LD_LIBRARY_PATH

julia_project_location="../../"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} julia --project="${julia_project_location}" progressive_growing_top_wikitext.jl \
  --dataset wikitext \
  --gpu 0 \
  --num-independent-clusters 200 \
  --num-init-clusters 2 \
  --num-final-clusters 4 \
  --fname-idx 4 \
  --num-tr-samples 20000 \
  --num-val-samples 5000 \
  --batch-size 256

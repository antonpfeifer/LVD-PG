#!/bin/usr/env bash
set -e
source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

export PYTHON=/scratch/tmp/mpfeife3/bachelorarbeit/miniconda/envs/lvd-pg/bin/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/julia:$LD_LIBRARY_PATH

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$PYTHON"
which python
which julia

julia -e 'using PyCall; println("PyCall Python: ", PyCall.pyprogramname); pyimport("faiss"); println("faiss ok")'

julia_project_location="../../"

CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 1 3200 200 "wikitext"
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 3201 6400 200 "wikitext"

set -a
source ../../.env
set +a

source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh

export JULIA_PROJECT=$CONDA_PREFIX/share/julia/environments/lvd-pg
export PYTHON=/scratch/tmp/mpfeife3/bachelorarbeit/miniconda/envs/lvd-pg/bin/python

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$PYTHON"
which python
which julia
echo "JULIA_PROJECT=$JULIA_PROJECT"

conda activate lvd-pg

LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python-jl progressive_growing_top.py -id --data-path "$DATA_PATH"

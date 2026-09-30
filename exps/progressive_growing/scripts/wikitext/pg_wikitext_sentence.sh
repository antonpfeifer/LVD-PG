#!/usr/bin/env bash
set -e
# Some login/slurm environments enable `set -u` (nounset). Conda's Julia
# activation hook may append to JULIA_DEPOT_PATH before it exists, which then
# aborts activation with: JULIA_DEPOT_PATH: unbound variable.
set +u
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-}"
export BASE_PATH="/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg"

# Guard against accidental direct submission (sbatch pg_wikitext_sentence.sh):
# without the palma wrapper this lands in the `normal` partition with ~2.5G
# and OOMs during Pkg.instantiate(). Submit via pg_wikitext_sentence_palma.sh.
_req_mem_digits="$(echo "${SLURM_MEM_PER_NODE:-}" | tr -cd '0-9')"
if [ -n "${_req_mem_digits}" ] && [ "${_req_mem_digits}" -gt 0 ] && [ "${_req_mem_digits}" -lt 16000 ]; then
    echo "ERROR: only ${_req_mem_digits}MB allocated (partition: ${SLURM_JOB_PARTITION:-unknown})." >&2
    echo "ERROR: submit via pg_wikitext_sentence_palma.sh (gpuh200mini, 128G) instead." >&2
    exit 1
fi
unset _req_mem_digits
source /scratch/tmp/mpfeife3/bachelorarbeit/miniconda/etc/profile.d/conda.sh
conda activate lvd-pg

export PYTHON=/scratch/tmp/mpfeife3/bachelorarbeit/miniconda/envs/lvd-pg/bin/python
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/julia:${LD_LIBRARY_PATH:-}"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$PYTHON"
which python
which julia

julia -e 'using PyCall; println("PyCall Python: ", PyCall.pyprogramname); pyimport("faiss"); println("faiss ok")'

JULIA_PROJ=$BASE_PATH

julia --project="${JULIA_PROJ}" -e 'using Pkg; println(Pkg.project().path)'
julia --project="${JULIA_PROJ}" -e "using Pkg; Pkg.activate(\"${JULIA_PROJ}\")"
julia --project="${JULIA_PROJ}" -e 'using Pkg; Pkg.instantiate()'

CUDA_VISIBLE_DEVICES=0 julia --project="${JULIA_PROJ}" "${BASE_PATH}/exps/progressive_growing/parallel_PG_sentence.jl" 1 100 400 "wikitext" || { echo "shard 1-100 failed" >&2; SHARD_FAIL=1; }
CUDA_VISIBLE_DEVICES=1 julia --project="${JULIA_PROJ}" "${BASE_PATH}/exps/progressive_growing/parallel_PG_sentence.jl" 101 200 400 "wikitext" || { echo "shard 101-200 failed" >&2; SHARD_FAIL=1; }
CUDA_VISIBLE_DEVICES=2 julia --project="${JULIA_PROJ}" "${BASE_PATH}/exps/progressive_growing/parallel_PG_sentence.jl" 201 300 400 "wikitext" || { echo "shard 201-300 failed" >&2; SHARD_FAIL=1; }
CUDA_VISIBLE_DEVICES=3 julia --project="${JULIA_PROJ}" "${BASE_PATH}/exps/progressive_growing/parallel_PG_sentence.jl" 301 400 400 "wikitext" || { echo "shard 301-400 failed" >&2; SHARD_FAIL=1; }

# `set -e` is kept for the setup phase above, but one failed shard must not
# skip the remaining shards (previously the first CUDA failure aborted the
# whole batch). Report a collective failure at the end instead.
exit "${SHARD_FAIL:-0}"

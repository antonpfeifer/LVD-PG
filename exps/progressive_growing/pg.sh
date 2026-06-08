#!/usr/bin/env bash
set -euo pipefail

export JULIA_DEPOT_PATH="$HOME/.julia:"
julia_project_location="../../"
#do global clustering and log firstly
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 1 1 20 "${@}"
#run multi-process to do progressive growing
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 1 5 20 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 6 10 20 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 11 15 20 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 16 20 20 "${@}" &
wait

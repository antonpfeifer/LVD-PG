#!/usr/bin/env bash
set -euo pipefail

export JULIA_DEPOT_PATH="$HOME/.julia:"
julia_project_location="../../"
#do global clustering and log firstly
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 1 1 400 "${@}"
#run multi-process to do progressive growing
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 1 100 400 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 101 200 400 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 201 300 400 "${@}" &
CUDA_VISIBLE_DEVICES=0 julia --project="${julia_project_location}" parallel_PG.jl 301 400 400 "${@}" &
wait

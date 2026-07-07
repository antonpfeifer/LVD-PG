using PyCall
using Printf
using Statistics: mean

cd(@__DIR__)

include("../../VariationalJuice.jl/src/VariationalJuice.jl")
include("../../VariationalJuice.jl/src-jl/LatentPCs.jl")

py"""
import numpy as _np

def _load_npy_slice(path, max_examples):
    a = _np.load(path, mmap_mode="r")
    if max_examples is None:
        return _np.asarray(a)
    n = min(int(max_examples), a.shape[0])
    return _np.asarray(a[:n])
"""

"""
    load_npy_data(path; max_examples=nothing)

Load a `.npy` dataset in the same row-major sample layout used by
`parallel_PG.jl`.  The returned value is a Julia `Array` whose first dimension
is the number of examples.
"""
function load_npy_data(path::AbstractString; max_examples=nothing)
    isfile(path) || error("Data file not found: $(path)")
    Array(py"_load_npy_slice"(path, max_examples))
end

"""
    load_pg_pc(path; multihead=true)

Load a PC produced by progressive growing.  Low-level PCs saved by
`write_mhpc` are stored as a summation over heads, so the default returns the
vector of head PCs expected by `CuMultiHeadBitsProbCircuit`.  Set
`multihead=false` for a normal single-root PC saved with `write(file, pc)`.
"""
function load_pg_pc(path::AbstractString; multihead::Bool=true)
    isfile(path) || error("PC file not found: $(path)")
    multihead ? read_mhpc(String(path)) : read(String(path), ProbCircuit)
end

function _assigned_head_mask(cluster_ids, num_examples::Integer, num_heads::Integer)
    length(cluster_ids) == num_examples || error("cluster_ids length $(length(cluster_ids)) does not match number of examples $(num_examples)")
    head_mask = zeros(Float32, num_examples, num_heads)
    for (i, cid) in enumerate(cluster_ids)
        1 <= cid <= num_heads || error("cluster id $(cid) at example $(i) is outside 1:$(num_heads)")
        head_mask[i, Int(cid)] = 1.0f0
    end
    head_mask
end

"""
    perplexity(pcs, data; batch_size=256, mode=:mixture, cluster_ids=nothing,
               num_tokens_per_example=size(data, 2))

Calculate perplexity for a progressive-growing PC on `data`.

For a multi-head PC (`pcs::Vector{<:ProbCircuit}`), `mode` controls how heads
are combined:

  * `:mixture`  – uniform mixture over all heads (default).
  * `:best`     – oracle assignment to the head with highest likelihood.
  * `:assigned` – use the provided 1-based `cluster_ids`, matching the hard
                  cluster evaluation in `parallel_PG.jl`.

The returned named tuple contains the perplexity, average negative log
likelihood per token (`nll_per_token`), average log likelihood per example, and
bits-per-token.
"""
function perplexity(
    pcs::Vector{<:ProbCircuit},
    data;
    batch_size::Integer=256,
    mode::Symbol=:mixture,
    cluster_ids=nothing,
    num_tokens_per_example::Real=size(data, 2),
)
    num_examples = size(data, 1)
    num_examples == 0 && error("Cannot compute perplexity of an empty dataset")
    num_tokens_per_example > 0 || error("num_tokens_per_example must be positive")

    data_gpu = cu(data)
    mhbpc = CuMultiHeadBitsProbCircuit(pcs)
    effective_batch_size = min(batch_size, num_examples)

    lls = if mode == :mixture
        # A dense all-ones mask makes the multi-head likelihood code compute
        # log(mean_h p_h(x)), i.e. a uniform mixture over heads.
        head_mask = CUDA.ones(Float32, num_examples, length(pcs))
        Array(loglikelihoods(mhbpc, data_gpu, head_mask; batch_size=effective_batch_size))
    elseif mode == :assigned
        cluster_ids === nothing && error("mode=:assigned requires cluster_ids")
        head_mask = cu(_assigned_head_mask(cluster_ids, num_examples, length(pcs)))
        Array(loglikelihoods(mhbpc, data_gpu, head_mask; batch_size=effective_batch_size))
    elseif mode == :best
        per_head_lls = Array(loglikelihoods(mhbpc, data_gpu, nothing; batch_size=effective_batch_size))
        vec(maximum(per_head_lls; dims=2))
    else
        error("Unknown mode $(mode). Use :mixture, :best, or :assigned.")
    end

    avg_ll_per_example = mean(lls)
    nll_per_token = -avg_ll_per_example / num_tokens_per_example
    (
        perplexity = exp(nll_per_token),
        nll_per_token = nll_per_token,
        avg_ll_per_example = avg_ll_per_example,
        bits_per_token = nll_per_token / log(2.0),
    )
end

function perplexity(
    pc::ProbCircuit,
    data;
    batch_size::Integer=256,
    num_tokens_per_example::Real=size(data, 2),
)
    num_examples = size(data, 1)
    num_examples == 0 && error("Cannot compute perplexity of an empty dataset")
    num_tokens_per_example > 0 || error("num_tokens_per_example must be positive")

    data_gpu = cu(data)
    bpc = CuBitsProbCircuit(pc)
    effective_batch_size = min(batch_size, num_examples)
    lls = Array(loglikelihoods_probcat(bpc, data_gpu; batch_size=effective_batch_size))

    avg_ll_per_example = mean(lls)
    nll_per_token = -avg_ll_per_example / num_tokens_per_example
    (
        perplexity = exp(nll_per_token),
        nll_per_token = nll_per_token,
        avg_ll_per_example = avg_ll_per_example,
        bits_per_token = nll_per_token / log(2.0),
    )
end

"""
    perplexity(pc_path, data; multihead=true, kwargs...)
    perplexity(pc_path, data_path; multihead=true, max_examples=nothing, kwargs...)

Convenience wrappers that load a saved PC (and optionally a `.npy` data file)
before evaluating perplexity.
"""
function perplexity(pc_path::AbstractString, data; multihead::Bool=true, kwargs...)
    pc = load_pg_pc(pc_path; multihead)
    perplexity(pc, data; kwargs...)
end

function perplexity(pc_path::AbstractString, data_path::AbstractString; multihead::Bool=true, max_examples=nothing, kwargs...)
    data = load_npy_data(data_path; max_examples)
    perplexity(pc_path, data; multihead, kwargs...)
end

function _parse_arg(flag, default, T)
    idx = findfirst(==(flag), ARGS)
    idx === nothing && return default
    idx == length(ARGS) && error("Missing value for $(flag)")
    T == String ? ARGS[idx + 1] : T == Symbol ? Symbol(ARGS[idx + 1]) : parse(T, ARGS[idx + 1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    pc_path = _parse_arg("--pc", "", String)
    data_path = _parse_arg("--data", "", String)
    isempty(pc_path) && error("Usage: julia perplexity.jl --pc PATH --data DATA.npy [--mode mixture|best|assigned] [--batch-size 256] [--tokens-per-example N] [--gpu 0]")
    isempty(data_path) && error("Usage: julia perplexity.jl --pc PATH --data DATA.npy [--mode mixture|best|assigned] [--batch-size 256] [--tokens-per-example N] [--gpu 0]")

    gpu = _parse_arg("--gpu", 0, Int)
    select_gpu(gpu)

    data = load_npy_data(data_path; max_examples=_parse_arg("--max-examples", nothing, Int))
    tokens_per_example = _parse_arg("--tokens-per-example", size(data, 2), Float64)
    result = perplexity(
        pc_path,
        data;
        multihead=_parse_arg("--multihead", true, Bool),
        mode=_parse_arg("--mode", :mixture, Symbol),
        batch_size=_parse_arg("--batch-size", 256, Int),
        num_tokens_per_example=tokens_per_example,
    )

    @printf("perplexity: %.6f\n", result.perplexity)
    @printf("nll/token:  %.6f\n", result.nll_per_token)
    @printf("bits/token: %.6f\n", result.bits_per_token)
end

include("split.jl")

struct EvaluationResult
    bpd::Split{Float32}
    perplexity::Split{Float32}
    mean_bpd::Split{Float32}
    per_cluster_ll::Array{Float32}
end

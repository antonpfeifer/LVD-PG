using Core: EvalInto
using PyCall
using NPZ
using Printf
using VariationalJuice

cd(@__DIR__)
include("model/split.jl")
include("model/evaluation_result.jl")
include("model/pg_config.jl")
push!(PyVector(pyimport("sys")["path"]), "./src")

py"""
from kmeans import train_kmeans_model, pred_kmeans_clusters
import numpy as _np

def subset_rows_1based(a, idx):
    # idx comes from Julia's findall, so it is 1-based.
    idx0 = _np.asarray(idx, dtype=_np.int64) - 1
    return a[idx0, :]
"""

np = pyimport("numpy")

function kmeans(trn_features, val_features, num_independent_clusters::Int, dataset, level, position::Int=nothing)
    # Perform global KMeans clustering
    cls_file_name = "temp/temp_$(dataset)/global_indep_cls/clusters_$(num_independent_clusters)_$(level)$(position != nothing ? "_pos$(position)" : "").npz"
    if !isdir("temp/temp_$(dataset)/global_indep_cls")
        mkpath("temp/temp_$(dataset)/global_indep_cls")
    end
    if !isfile(cls_file_name)
        print("> Global clustering into $(num_independent_clusters) clusters... ")
        t = @elapsed begin
            centroids = py"train_kmeans_model"(trn_features, num_independent_clusters)
            cls_ids_trn = Int64.(py"pred_kmeans_clusters"(centroids, trn_features))
            cls_ids_val = Int64.(py"pred_kmeans_clusters"(centroids, val_features))
        end
        println(@sprintf("done (%.2fs)", t))
        GC.gc()

        NPZ.npzwrite(cls_file_name, Dict("sentence_cls_ids_trn" => cls_ids_trn, "sentence_cls_ids_val" => cls_ids_val))
    else
        println("> Loaded global cluster ids")
        data = NPZ.npzread(cls_file_name)
        cls_ids_trn = data["sentence_cls_ids_trn"]
        cls_ids_val = data["sentence_cls_ids_val"]
    end
    return cls_ids_trn, cls_ids_val
end

function main(; dataset, start_cid, end_cid, pg_config::ProgressiveGrowingConfig)
    data_dir = "../LVD_for_wikitext/data/data_$(dataset)"
    trn_data = Array{Int32}(np.load(joinpath(data_dir, "data_trn.npy")))
    val_data = Array{Int32}(np.load(joinpath(data_dir, "data_val.npy")))
    token_trn_features = Array{Float32}(np.load(joinpath(data_dir, "tokenfeat_trn.npy"), mmap_mode="r"))
    token_val_features = Array{Float32}(np.load(joinpath(data_dir, "tokenfeat_val.npy"), mmap_mode="r"))
    sentence_trn_features = Array{Float32}(np.load(joinpath(data_dir, "sentencefeat_trn.npy"), mmap_mode="r"))
    sentence_val_features = Array{Float32}(np.load(joinpath(data_dir, "sentencefeat_val.npy"), mmap_mode="r"))

    @assert size(trn_data, 1) == size(token_trn_features, 1) == size(sentence_trn_features, 1) "the number of training examples (sentences) for sentence and token features, as well as raw training data should be the same"
    @assert size(val_data, 1) == size(token_val_features, 1) == size(sentence_val_features, 1) "the number of validation examples (sentences) for sentence and token features, as well as raw validation data should be the same"

    @assert size(trn_data, 2) == size(token_trn_features, 2) "the number of token positions (max sentence size) should be the same for raw token data and annotated token data"

    sentence_cid_trn, sentence_cid_val = kmeans(sentence_trn_features, sentence_val_features, pg_config.num_sentence_clusters, dataset, "sentence")

    num_token_positions = size(trn_data, 2) # aka max_sentence_size

    task_identifier = "pos$(num_token_positions)_token$(pg_config.num_token_clusters)_indep$(pg_config.num_sentence_clusters)_init$(pg_config.num_init_clusters)_final$(pg_config.num_final_clusters)"
    ll_file_name = "temp/temp_$(dataset)/logs/$(task_identifier)_parallel.log"
    if !isdir("temp/temp_$(dataset)/logs")
        mkpath("temp/temp_$(dataset)/logs")
    end
    total_trn_bpd = 0.0
    total_val_bpd = 0.0
    val_bpd_count = 0
    for cid = start_cid:end_cid # iteration durch satz-cluster
        println(">>> Progressive growing #$(cid) <<<")
        trn_filter = (sentence_cid_trn .== cid) # sentence indixes with current sentence cid
        val_filter = (sentence_cid_val .== cid)

        trn_weight = size(trn_data[trn_filter, :], 1) # number of training samples to be considered
        val_weight = size(val_data[val_filter, :], 1)

        println("tr_weight: $(trn_weight) ts_weight: $(val_weight)")

        if trn_weight == 0
            println(">>> Empty cluster #$(cid); skipping <<<")
            continue
        end


        # CLUSTER TOKENS

        token_cids_trn = zeros(Int32, trn_weight, num_token_positions)
        token_cids_val = zeros(Int32, val_weight, num_token_positions)

        for pos = 1:num_token_positions
            tokens_cid_pos_trn_features = token_trn_features[trn_filter, pos, :] # tokens from sentence cid at pos
            tokens_cid_pos_val_features = token_val_features[val_filter, pos, :]

            pos_cids_trn, pos_cids_val = kmeans(tokens_cid_pos_trn_features, tokens_cid_pos_val_features, pg_config.num_token_clusters, dataset, "token", pos)

            token_cids_trn[:, pos] .= pos_cids_trn
            token_cids_val[:, pos] .= pos_cids_val
        end


        effective_final_clusters = max(1, min(pg_config.num_final_clusters, trn_weight))
        final_pc_fname = "temp/temp_$(dataset)/final_pcs/$(task_identifier)/$(cid)/final_pc_$(effective_final_clusters).jpc"
        if isfile(final_pc_fname)
            println(">>> Existing mhpc #$(cid) <<<")
            continue
        end

        trn_idx = findall(trn_filter)
        val_idx = findall(val_filter)

        # yz_*_features are numpy memmaps/PyObjects. PyCall cannot index them with
        # Julia BitVectors, so index in Python with explicit row ids, then convert
        # the per-cluster subset to a normal Julia Array. These subsets are small.
        trn_features_subset = Array(py"subset_rows_1based"(sentence_trn_features, trn_idx))
        val_features_subset = Array(py"subset_rows_1based"(sentence_val_features, val_idx))

        trn_bpd, val_bpd = progressive_growing(
            dataset_label=dataset,
            token_cids=Split{AbstractArray{Int32, 2}}(token_cids_trn, token_cids_val),
            raw_data=Split{AbstractArray{Int32, 2}}(trn_data[trn_filter, :], val_data[val_filter, :]),
            sentence_features=Split{AbstractArray{Float32, 2}}(trn_features_subset, val_features_subset),
            global_task_id=cid,
            task_identifier=task_identifier,
            config=pg_config
        )
        total_trn_bpd += trn_bpd
        mean_trn_bpd = total_trn_bpd / (cid - start_cid + 1)
        if !isnan(val_bpd)
            total_val_bpd += val_bpd
            val_bpd_count += 1
        end
        mean_val_bpd = val_bpd_count > 0 ? total_val_bpd / val_bpd_count : NaN


        print(@sprintf("cid: %d trn: %.4f; val: %.4f mean(%.4f,%.4f) \n", cid, trn_bpd, val_bpd, mean_trn_bpd, mean_val_bpd))
        open(ll_file_name, "a") do io
            write(io, @sprintf("cid: %d trn: %.4f %d; val: %.4f %d mean(%.4f,%.4f)\n", cid, trn_bpd, trn_weight, val_bpd, val_weight, mean_trn_bpd, mean_val_bpd))
        end
    end
end

function fit_token_emissions(trn_data, trn_token_cids; num_token_clusters::Int,
        vocab_size::Int=151936, alpha::Float64=1.0,
        backoff::Union{Nothing,AbstractVector}=nothing,
        pad_id::Union{Nothing,Integer}=nothing)
    @assert alpha >= 0.0 "emission smoothing alpha must be >= 0, got $(alpha)"
    positions::Int = size(trn_data, 2)
    @assert size(trn_token_cids, 2) == positions "raw training data should have the same amount of max token count in a sentence as token cluster training data"
    @assert size(trn_token_cids, 1) == size(trn_data, 1) "raw training data should have the same amount of examples as token cluster training data"

    # OPTIONAL BACKOFF DISTRIBUTION for smoothing; if not provided, compute from training data
    backoff_vec::Vector{Float32} = if backoff !== nothing
        @assert length(backoff) == vocab_size "backoff length $(length(backoff)) != vocab_size $(vocab_size)"
        b = Float32.(backoff)
        sb = sum(b)
        sb > 0 ? b ./ sb : fill(Float32(1 / vocab_size), vocab_size)
    else
        glob = zeros(Float64, vocab_size) # global token counts across all positions and clusters
        total = 0.0
        # iterate through training data and count occurrences of each token
        @inbounds for i in 1:size(trn_data, 1), pos in 1:positions
            tok = Int(trn_data[i, pos])
            (pad_id !== nothing && tok == pad_id) && continue
            idx = tok + 1 # teacher ids are 0-based -> Julia 1-based
            (1 <= idx <= vocab_size) || continue
            glob[idx] += 1.0
            total += 1.0
        end
        total > 0 ? Float32.(glob ./ total) : fill(Float32(1 / vocab_size), vocab_size)
    end

    # KMeans ids are 1-based (pred_kmeans_clusters adds +1); tolerate 0-based inputs.
    cid_zero_based::Bool = minimum(trn_token_cids) == 0

    # count occurrences of each token in each cluster at each position -> categorial distributions
    token_emissions = Array{Float32,3}(undef, positions, num_token_clusters, vocab_size)
    counts = Vector{Float32}(undef, vocab_size)
    alpha32 = Float32(alpha)
    for pos = 1:positions
        for cid = 1:num_token_clusters
            fill!(counts, 0.0f0)
            n = 0
            @inbounds for i in 1:size(trn_data, 1)
                c = Int(trn_token_cids[i, pos]) + (cid_zero_based ? 1 : 0)
                c == cid || continue
                tok = Int(trn_data[i, pos])
                (pad_id !== nothing && tok == pad_id) && continue
                idx = tok + 1
                (1 <= idx <= vocab_size) || continue
                counts[idx] += 1.0f0
                n += 1
            end
            out = @view token_emissions[pos, cid, :]
            if n == 0
                out .= backoff_vec
            elseif alpha32 == 0.0f0
                # Pure MLE; floor zeros so later log() stays finite.
                s = Float32(n)
                @inbounds for v in 1:vocab_size
                    out[v] = counts[v] > 0 ? counts[v] / s : 1.0f-9
                end
                out ./= sum(out)
            else
                denom = Float32(n) + alpha32
                @inbounds for v in 1:vocab_size
                    out[v] = (counts[v] + alpha32 * backoff_vec[v]) / denom
                end
            end
        end
    end
    token_emissions
end


function compile_token_view(token_emissions::AbstractArray{Float32, 3}, pcs)
    positions, token_clusters, vocab_size = size(token_emissions)
    map(pcs) do pc
        by_var = Dict{Int,Vector}()
        foreach(pc) do n
            isinput(n) && push!(get!(by_var, randvar(n), []), n)
        end
        hclt_states_counts = [length(by_var[pos]) for pos in 1:positions]
        @assert all(==(hclt_states_counts[1]), hclt_states_counts) "head has varying H per position: $(hclt_states_counts)"
        hclt_states_count = hclt_states_counts[1]
        view = Array{Float32,3}(undef, hclt_states_count, positions, vocab_size)
        for pos in 1:positions
            cluster_emissions = hcat([exp.(dist(l).logps) for l in by_var[pos]]...)
            view[:, pos, :] .= cluster_emissions' * token_emissions[pos, :, :] # add across all token clusters to get the probability distribution for individual tokens
        end
        view
    end
end

function get_token_pcs(pcs, positions::Int, token_views)
    map(enumerate(pcs)) do (hi, pc)
        token_pc = deepcopy(pc)
        leaf_nodes = Dict{Int,Vector}()
        foreach(token_pc) do n
            isinput(n) && push!(get!(leaf_nodes, randvar(n), []), n)
        end
        for pos in 1:positions
            @assert length(leaf_nodes[pos]) == size(token_views[hi], 1) "somhow the pc's leaf node distribution has a different amount of latent variables than the token view"
            for (h, leaf) in enumerate(leaf_nodes[pos])
                leaf.dist = Categorical(Vector{Float32}(log.(token_views[hi][h, pos, :])))
            end
        end
        token_pc
    end
end

function evaluate_pcs(mhbpc, data_gpu::Split, effective_batch_size::Split{Int}, cls_ids::Split{Vector{Int64}}, num_vars, num_clusters, val_examples)
    per_sample_lls = Array(loglikelihoods(mhbpc, data_gpu.trn, nothing; batch_size=effective_batch_size.trn))
    mean_trn_bpd = -mean(per_sample_lls) / log(2.0) / num_vars
    best_lls, _ = findmax(per_sample_lls; dims=2)
    min_trn_bpd = -mean(best_lls) / log(2.0) / num_vars

    per_cluster_weight = map(1:num_clusters) do idx
        size(per_sample_lls[cls_ids.trn.==idx, idx], 1)
    end
    per_cluster_ll = map(1:num_clusters) do idx
        per_cluster_weight[idx] == 0 ? zero(Float32) : mean(per_sample_lls[cls_ids.trn.==idx, idx])
    end
    per_cluster_bpd = -per_cluster_ll / log(2.0) / num_vars
    trn_bpd = sum(per_cluster_bpd .* per_cluster_weight) / sum(per_cluster_weight)
    trn_perplexity = exp(trn_bpd * log(2.0))

    if val_examples > 0
        per_sample_lls_val = Array(loglikelihoods(mhbpc, data_gpu.val, nothing; batch_size=effective_batch_size.val))
        mean_val_bpd = -mean(per_sample_lls_val) / log(2.0) / num_vars
        best_lls_val, _ = findmax(per_sample_lls_val; dims=2)
        min_val_bpd = -mean(best_lls_val) / log(2.0) / num_vars


        per_cluster_weight_val = map(1:num_clusters) do idx
            sum(cls_ids.val .== idx)
        end

        per_cluster_ll_val = map(1:num_clusters) do idx
            if per_cluster_weight_val[idx] > 0
                mean(per_sample_lls_val[cls_ids.val.==idx, idx])
            else
                zero(Float32)
            end
        end

        per_cluster_bpd_val = -per_cluster_ll_val / log(2.0) / num_vars
        val_bpd = sum(per_cluster_bpd_val .* per_cluster_weight_val) / sum(per_cluster_weight_val)
        val_perplexity = exp(val_bpd * log(2.0))
    else
        mean_val_bpd = NaN
        val_bpd = NaN
        val_perplexity = NaN
    end
    EvaluationResult(Split(trn_bpd, val_bpd), Split(trn_perplexity, val_perplexity), Split(mean_trn_bpd, mean_val_bpd), per_cluster_ll)
end

function log_evaluation(eval::EvaluationResult, mhbpc, num_clusters::Integer, num_heads::Integer; filename::String)
    mkpath(dirname(filename))
    println("  - Weighted bpd: ($(eval.bpd.trn),$(eval.bpd.val))")
    println("  - Weighted perplexity: ($(eval.perplexity.trn),$(eval.perplexity.val))")
    println("  - Overall average bpd: ($(eval.mean_bpd.trn),$(eval.mean_bpd.val))")
    println("  - Number of nodes: $(length(mhbpc.bpc.nodes) - 1)")
    println("  - Number of edges: $(length(mhbpc.bpc.edge_layers_up.vectors) - num_clusters)")

    open(filename, "a") do io
        write(io, @sprintf("trn: %.4f(%.4f) val: %.4f(%.4f) n_cls:%d \n", eval.bpd.trn, eval.perplexity.trn, eval.bpd.val, eval.perplexity.val, num_heads))
    end
end

function progressive_growing(;
    dataset_label, token_cids::Split{AbstractArray{Int32, 2}}, raw_data::Split{AbstractArray{Int64, 2}}, sentence_features::Split{AbstractArray{Float32, 2}},
    global_task_id, task_identifier, config::ProgressiveGrowingConfig
)
    # Load tokenizer/model metadata written by the earlier Python preprocessing step.
    metadata_path = joinpath("..", "LVD_for_wikitext", "data", "data_$(dataset_label)", "model_metadata.json")
    @assert isfile(metadata_path) "missing model metadata file: $(metadata_path)"
    metadata_file = pyimport("builtins").open(metadata_path, "r", encoding="utf-8")
    metadata = pyimport("json").load(metadata_file)
    metadata_file.close()

    pad_token_id = Int(metadata["pad_token_id"])
    vocab_size = Int(metadata["vocab_size"])

    @assert size(token_cids.trn, 2) == size(raw_data.trn, 2) "token cluster assignments and raw token data do not have the same number of positions (max sentence size)"

    num_examples = Split{Int}(size(token_cids.trn, 1), size(token_cids.val, 1))
    num_token_positions::Int = size(token_cids.trn, 2)

    # KMeans ids are 1-based; Categorical leaves, Chow-Liu MI, and GPU kernels are 0-based.
    trn_token_cids = Int32.(token_cids.trn) .- Int32(1)
    val_token_cids = Int32.(token_cids.val) .- Int32(1)
    if !isempty(trn_token_cids)
        @assert minimum(trn_token_cids) >= 0 && maximum(trn_token_cids) < config.num_token_clusters "token cluster ids outside 0:$(config.num_token_clusters - 1)"
    end
    if !isempty(val_token_cids)
        @assert minimum(val_token_cids) >= 0 && maximum(val_token_cids) < config.num_token_clusters "val token cluster ids outside 0:$(config.num_token_clusters - 1)"
    end

    # Some global positionwise clusters can be extremely small. FAISS KMeans
    # requires n_examples >= n_clusters, so cap local cluster counts per task.
    num_init_clusters = max(1, min(config.num_init_clusters, num_examples.trn))
    num_final_clusters = max(1, min(config.num_final_clusters, num_examples.trn))

    effective_batch_size = Split(min(config.batch_size, num_examples.trn), min(128, max(num_examples.val, 1)))

    token_cids_gpu = Split(cu(trn_token_cids), cu(val_token_cids))

    if !isdir("temp/temp_$(dataset_label)/init_pcs/$(task_identifier)")
        mkpath("temp/temp_$(dataset_label)/init_pcs/$(task_identifier)")
    end

    if !isdir("temp/temp_$(dataset_label)/final_pcs/$(task_identifier)")
        mkpath("temp/temp_$(dataset_label)/final_pcs/$(task_identifier)")
    end


    if !isdir("temp/temp_$(dataset_label)/logs/$(task_identifier)")
        mkpath("temp/temp_$(dataset_label)/logs/$(task_identifier)")
    end

    grow_ll_file_name = "temp/temp_$(dataset_label)/logs/$(task_identifier)/$(global_task_id).log"

    # Perform initial KMeans clustering
    # print("> Clustering all samples into $(num_init_clusters) clusters... ")
    t = @elapsed begin
        centroids = py"train_kmeans_model"(sentence_features.trn, config.num_init_clusters)
        trn_cls_ids = Int64.(py"pred_kmeans_clusters"(centroids, sentence_features.trn))
        val_cls_ids = Int64.(py"pred_kmeans_clusters"(centroids, sentence_features.val))
    end
    # println(@sprintf("done (%.2fs)", t))
    GC.gc()

    # Generate initial structure
    base_dir = "temp/temp_$(dataset_label)/init_pcs/$(task_identifier)/$(global_task_id)"
    if !isdir(base_dir)
        mkpath(base_dir)
    end
    base_dir1 = "temp/temp_$(dataset_label)/final_pcs/$(task_identifier)/$(global_task_id)"
    if !isdir(base_dir1)
        mkpath(base_dir1)
    end
    init_pc_fname = joinpath(base_dir, "init_pc_$(config.num_init_clusters).jpc")
    # final_pc_fname = joinpath(base_dir1, "final_pc_$(num_final_clusters).jpc")

    if !isfile(init_pc_fname)
        println("> Constructing initial multi-headed PC...")
        token_cid_datasets = []
        for cid = 1:config.num_init_clusters
            token_cid_dataset = trn_token_cids[trn_cls_ids.==cid, :]
            # Convert to CPU Array explicitly just in case to avoid passing mixed arrays
            push!(token_cid_datasets, Array(token_cid_dataset))
        end
        println("dataset: $(dataset_label)")
        pcs = joined_hclt(token_cid_datasets, config.num_hclt_latents; num_cats=config.num_token_clusters, input_type=Categorical)
        pcs = pcs[1:config.num_init_clusters]
        init_parameters(pcs; perturbation=0.4)

        write_mhpc(init_pc_fname, pcs)
    else
        println("> Loaded initial multi-headed PC")
        pcs = read_mhpc(init_pc_fname)
    end

    ##### Main loop #####
    mean_trn_bpd = 0.0
    mean_val_bpd = 0.0
    trn_bpd = 0.0
    val_bpd = 0.0


    ##choose which ckpt to save
    if num_final_clusters == 20
        c = [5, 10, 15, 20]
    elseif num_final_clusters == 10
        c = [4, 7, 10]
    elseif num_final_clusters == 5
        c = [3, 4, 5]
    elseif num_final_clusters == 4
        c = [4]
    elseif num_final_clusters == 1
        c = [1]
    else
        # Happens for tiny global clusters after capping num_final_clusters.
        c = [num_final_clusters]
    end


    final_pc_fnames = []
    for ci in c
        final_pc_fname = joinpath(base_dir1, "final_pc_$(ci).jpc")
        push!(final_pc_fnames, final_pc_fname)
    end
    final_pc_fname = final_pc_fnames[1]

    for iter = config.num_init_clusters:2*num_final_clusters
        println("==== Iteration $(iter) ====")

        num_clusters = length(pcs)

        ## Step 1: train the multi-head PC
        println("> Training multi-head PC...")
        mhbpc = CuMultiHeadBitsProbCircuit(pcs)
        trn_head_mask = zeros(Float32, num_examples.trn, num_clusters)
        ids = [CartesianIndex(i, j) for (i, j) in zip(collect(1:num_examples.trn), trn_cls_ids)]
        trn_head_mask[ids] .= one(Float32)
        trn_head_mask_gpu = cu(trn_head_mask)

        lls = mini_batch_em_for_multihead_pc(
            mhbpc, token_cids_gpu.trn, trn_head_mask_gpu, 50;
            batch_size=effective_batch_size.trn, pseudocount=0.1, soft_reg=0.0, soft_reg_width=3,
            param_inertia=0.9, param_inertia_end=0.99
        )
        update_parameters(mhbpc)

        mean_bpd = if num_examples.val > 0
            per_sample_lls = Array(loglikelihoods(mhbpc, token_cids_gpu.val, nothing; batch_size=effective_batch_size.val))
            -mean(per_sample_lls) / log(2.0) / num_token_positions
        else
            NaN
        end

        println("  - Train bpd: $(-lls[end] / log(2.0) / num_token_positions)")
        println("  - Test  bpd: $mean_bpd")
        println("  - Number of nodes: $(length(mhbpc.bpc.nodes) - 1)")
        println("  - Number of edges: $(length(mhbpc.bpc.edge_layers_up.vectors) - num_clusters)")

        ## Step 2: prune multi-head PC
        print("> Pruning multi-head PC...")
        t = @elapsed pcs = prune_pc(pcs, token_cids_gpu.trn, trn_head_mask_gpu; batch_size=effective_batch_size.trn, prune_threshold=config.prune_threshold, mhbpc)
        println(@sprintf("done (%.2fs)", t))

        ## Step 3: evaluate and re-assign cluster ids
        mhbpc = CuMultiHeadBitsProbCircuit(pcs)

        eval::EvaluationResult = evaluate_pcs(mhbpc, token_cids_gpu, effective_batch_size, Split(trn_cls_ids, val_cls_ids), num_token_positions, num_clusters, num_examples.val)

        log_evaluation(eval, mhbpc, num_clusters, length(pcs); filename=grow_ll_file_name)
        trn_bpd, val_bpd = eval.bpd.trn, eval.bpd.val

        if length(c) == 1
            if length(pcs) >= c[1] && !isfile(final_pc_fname)
                write_mhpc(final_pc_fname, pcs)
                break
            end
        else
            for i = 2:length(c)-1
                if length(pcs) >= c[i] && length(pcs) < c[i+1]
                    final_pc_fname = final_pc_fnames[i]
                    break
                end
            end

            if length(pcs) >= c[end]
                final_pc_fname = final_pc_fnames[end]
            end

            if length(pcs) >= c[1]
                if !isfile(final_pc_fname)
                    write_mhpc(final_pc_fname, pcs)
                end
            end

            if length(pcs) >= c[end]
                for final_pc_fname in final_pc_fnames
                    if !isfile(final_pc_fname)
                        write_mhpc(final_pc_fname, pcs)
                    end
                end
                break
            end
        end

        reassign_lls = Array(loglikelihoods(mhbpc, token_cids_gpu.trn, nothing; batch_size=effective_batch_size.trn))
        reassign_lls[ids] .+= 1.0 # Reluctant to switch cluster id
        _, mxcls = findmax(reassign_lls; dims=2)
        trn_cls_ids = map(id -> id.I[2], mxcls[:, 1])


        ## Step 4: decide clusters to grow
        # Recompute cluster weights after the reassignment above.  The weights
        # computed for reporting still refer to the previous assignments; using
        # them here can select clusters that became empty, which then gives a
        # zero grow batch size and crashes in grow_heads_by_flows.
        current_cluster_weight = map(1:num_clusters) do idx
            sum(trn_cls_ids .== idx)
        end
        grow_cls_true = Int[]
        sorted_lls = sortperm(eval.per_cluster_ll)
        thr = Int(round(num_examples.trn * 0.4))
        cnt = 0
        for cluster in sorted_lls
            if current_cluster_weight[cluster] == 0 || isnan(eval.per_cluster_ll[cluster])
                continue
            end
            push!(grow_cls_true, cluster)
            cnt += current_cluster_weight[cluster]
            if cnt > thr
                break
            end
        end

        if isempty(grow_cls_true)
            println("No non-empty clusters selected for growing; stopping progressive growing.")
            for final_pc_fname in final_pc_fnames
                if !isfile(final_pc_fname)
                    write_mhpc(final_pc_fname, pcs)
                end
            end
            break
        end

        if num_final_clusters <= 5
            target_n_clusters = length(grow_cls_true) + 1
        else
            thr1 = cnt ÷ 8000 #if dataset == "CIFAR10": 2500
            rand_min = length(grow_cls_true) + 1
            rand_max = min(num_final_clusters, 2 * length(grow_cls_true), thr1)
            if rand_min > rand_max
                print("\n thr1:", thr1, "\n")
                for final_pc_fname in final_pc_fnames
                    if !isfile(final_pc_fname)
                        write_mhpc(final_pc_fname, pcs)
                    end
                end
                break
            end
            target_n_clusters = rand_min + Int(round(rand() * (rand_max - rand_min)))
        end

        grow_n_clusters = target_n_clusters - length(grow_cls_true)
        grow_cls = grow_cls_true[1:grow_n_clusters]


        ## Step 5-1: grow the multi-head PC
        print("> Growing the multi-head PC...")
        t = @elapsed pcs = begin
            filter = zeros(Bool, num_examples.trn)
            for cluster in grow_cls
                filter .|= (trn_cls_ids .== cluster)
            end
            num_grow_examples = sum(filter)
            if num_grow_examples == 0
                error("Selected grow clusters $(grow_cls) contain no training examples after reassignment.")
            end
            head_mask = zeros(Float32, num_examples.trn, num_clusters)
            ids = [CartesianIndex(i, j) for (i, j) in zip(collect(1:num_examples.trn), trn_cls_ids)]
            head_mask[ids] .= one(Float32)
            grow_batch_size = min(effective_batch_size.trn, num_grow_examples)
            pcs = grow_heads_by_flows(
                pcs, token_cids_gpu.trn[filter, :], cu(head_mask[filter, :]);
                sigma=0.2, node_selection_method="percentage",
                node_selection_args=Dict("grow_frac" => max(grow_n_clusters / (grow_n_clusters + num_clusters), 0.2)),
                batch_size=grow_batch_size
            )
            @assert length(pcs) == num_clusters + grow_n_clusters
            pcs
        end
        println(@sprintf("done (%.2fs)", t))

        ## Step 5-2: update cluster ids
        all_trn_features = []
        old_centroids = []
        all_trn_data = []
        print("> Updating cluster ids...\n")
        for (i, cluster) in enumerate(grow_cls_true)
            trn_filter = (trn_cls_ids .== cluster)
            if i == 1
                all_trn_data = raw_data.trn[trn_filter, :]
                all_trn_features = sentence_features.trn[trn_filter, :]
                old_centroids = mean(sentence_features.trn[trn_filter, :], dims=1)
            else
                all_trn_data = cat(all_trn_data, raw_data.trn[trn_filter, :], dims=1)
                all_trn_features = cat(all_trn_features, sentence_features.trn[trn_filter, :], dims=1)
                old_centroids = cat(old_centroids, mean(sentence_features.trn[trn_filter, :], dims=1), dims=1)
            end
        end

        centroids = py"train_kmeans_model"(all_trn_features, target_n_clusters, centroids=old_centroids)
        trn_filter = zeros(Bool, num_examples.trn)
        val_filter = zeros(Bool, num_examples.val)
        for cluster in grow_cls_true
            trn_filter .|= (trn_cls_ids .== cluster)
            val_filter .|= (val_cls_ids .== cluster)
        end
        cls_ids_trn = Int64.(py"pred_kmeans_clusters"(centroids, sentence_features.trn[trn_filter, :]))
        cls_ids_val = Int64.(py"pred_kmeans_clusters"(centroids, sentence_features.val[val_filter, :]))
        GC.gc()
        for j = 1:target_n_clusters
            if j <= length(grow_cls_true)
                @views trn_cls_ids[trn_filter][cls_ids_trn.==j] .= grow_cls_true[j]
                @views val_cls_ids[val_filter][cls_ids_val.==j] .= grow_cls_true[j]
            else
                @views trn_cls_ids[trn_filter][cls_ids_trn.==j] .= j + num_clusters - length(grow_cls_true)
                @views val_cls_ids[val_filter][cls_ids_val.==j] .= j + num_clusters - length(grow_cls_true)
            end
        end
    end

    # Fit once at the end: phi depends only on (trn_data, trn_token_cids), so this is
    # identical to fitting before the loop but keeps ~8GB out of peak loop memory.
    token_emissions = fit_token_emissions(raw_data.trn, trn_token_cids;
        num_token_clusters=config.num_token_clusters, vocab_size=vocab_size, alpha=config.token_emission_alpha, pad_id=pad_token_id)
    token_views = compile_token_view(token_emissions, pcs)

    data_gpu = Split(cu(raw_data.trn), cu(raw_data.val))
    token_pcs = get_token_pcs(pcs, num_token_positions, token_views)
    mhbpc_tok = CuMultiHeadBitsProbCircuit(token_pcs)

    num_tok_examples = Split(size(raw_data.trn, 1), size(raw_data.val, 1))

    eval_tok = evaluate_pcs(mhbpc_tok, data_gpu, effective_batch_size, Split(trn_cls_ids, val_cls_ids), num_token_positions, length(pcs), num_tok_examples.val)

    log_evaluation(eval_tok, mhbpc_tok, length(pcs), length(token_pcs); filename="temp/temp_$(dataset_label)/logs/$(task_identifier)/tokens/$(global_task_id).log")

    trn_bpd, val_bpd
end

start_cid = parse(Int, ARGS[1])
end_cid = parse(Int, ARGS[2])
num_independent_clusters = parse(Int, ARGS[3])
dataset = ARGS[4]
println("dataset: $(dataset)")

num_token_clusters = 200
num_sentence_clusters = 400
num_hclt_latents = 16
num_init_clusters = 2
num_final_clusters = 4
batch_size = 256
max_grow_frac = 0.4
prune_threshold = 1e-4
token_emission_alpha = 1.0

main(; dataset=dataset,
    start_cid=start_cid,
    end_cid=end_cid,
    pg_config=ProgressiveGrowingConfig(
        num_token_clusters,
        num_sentence_clusters,
        num_hclt_latents,
        num_init_clusters,
        num_final_clusters,
        batch_size,
        max_grow_frac,
        prune_threshold,
        token_emission_alpha
    ))

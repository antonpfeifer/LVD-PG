using PyCall
using Printf
using Statistics: mean

cd(@__DIR__)

include("../../VariationalJuice.jl/src/VariationalJuice.jl")
include("../../VariationalJuice.jl/src-jl/LatentPCs.jl")

np = pyimport("numpy")

const DEFAULT_DATA_ROOT = normpath(joinpath(@__DIR__, "..", "progressive_growing", "data"))

py"""
import numpy as _np

def load_wikitext_slice(path, n):
    a = _np.load(path, mmap_mode="r")
    n = min(int(n), a.shape[0])
    return _np.asarray(a[:n])
"""

function chain_clt_edges(num_vars::Integer)
    [(i, i + 1) for i = 1:num_vars-1]
end

function load_wikitext_data(dataset::String, split::String, num_samples::Integer; data_root::String = DEFAULT_DATA_ROOT)
    data_path = joinpath(data_root, "data_$(dataset)", "data_$(split).npy")
    if !isfile(data_path)
        error("Wikitext data file not found: $(data_path). Generate it with exps/LVD_for_wikitext/get_data_for_PG.py or pass --data-root pointing to the directory containing data_$(dataset)/.")
    end
    data = Array(py"load_wikitext_slice"(data_path, num_samples))
    UInt8.(mod.(data, 256))
end

function collect_position_pcs(base_dir, task_identifier, fname_idx, num_positions, num_independent_clusters)
    position_pcs = Vector{Vector{ProbCircuit}}(undef, num_positions)
    max_position_heads = 0

    for pos = 1:num_positions
        pcs_for_pos = ProbCircuit[]
        for local_cid = 1:num_independent_clusters
            cluster_id = (pos - 1) * num_independent_clusters + local_cid
            final_pc_fname = joinpath(base_dir, "final_pcs", task_identifier, string(cluster_id), "final_pc_$(fname_idx).jpc")
            if !isfile(final_pc_fname)
                println("Warning: file $(final_pc_fname) not found. Skipping cluster $(cluster_id).")
                continue
            end
            append!(pcs_for_pos, read_mhpc(final_pc_fname))
        end
        position_pcs[pos] = pcs_for_pos
        max_position_heads = max(max_position_heads, length(pcs_for_pos))
        println(@sprintf("  - position %2d: %d low-level heads", pos, length(pcs_for_pos)))
    end

    max_position_heads == 0 && error("No low-level PCs found under $(joinpath(base_dir, "final_pcs", task_identifier)).")
    position_pcs, max_position_heads
end

function build_top_level_pseudo_data(data, position_pcs, num_hidden_cats; batch_size = 256)
    num_examples, num_positions = size(data)
    pseudo = fill(Float32(-1.0f30), num_examples, num_positions, num_hidden_cats)

    for pos = 1:num_positions
        pcs = position_pcs[pos]
        if isempty(pcs)
            println("Warning: no PCs for position $(pos); leaving pseudo likelihoods at -1e30.")
            continue
        end

        mhbpc = CuMultiHeadBitsProbCircuit(pcs)
        pos_data = cu(reshape(data[:, pos], :, 1))
        lls = Array(loglikelihoods(mhbpc, pos_data, nothing; batch_size = batch_size))
        pseudo[:, pos, 1:length(pcs)] .= Float32.(lls)
        CUDA.unsafe_free!(pos_data)
        GC.gc()
        CUDA.reclaim()

        min_bpd = -maximum(lls) / log(2.0)
        mean_bpd = -mean(lls) / log(2.0)
        max_bpd = -minimum(lls) / log(2.0)
        println(@sprintf("  - completed position %2d - low-level min/mean/max bpd: %.2f/%.2f/%.2f", pos, min_bpd, mean_bpd, max_bpd))
    end

    pseudo
end

function train_pg_top_level_wikitext(; dataset = "wikitext", data_root = DEFAULT_DATA_ROOT, fname_idx = 4,
        num_independent_clusters = 200, num_init_clusters = 2, num_final_clusters = 4,
        num_latents = 64, num_tr_samples = 20_000, num_val_samples = 5_000,
        batch_size = 256, num_epochs1 = 10, num_epochs2 = 10,
        pseudocount = 0.1, param_inertia1 = 0.9, param_inertia2 = 0.99,
        param_inertia3 = 0.999, gpu = 0)

    select_gpu(gpu)

    note = "id"
    base_dir = "temp/temp_$(dataset)"
    task_identifier = "$(note)_poswise_cat_l2_id$(num_independent_clusters)_init$(num_init_clusters)_final$(num_final_clusters)"
    ll_file_name = joinpath(base_dir, "logs", "$(task_identifier)_parallel.log")
    top_level_dir = joinpath(base_dir, "top_level_pcs")
    mkpath(top_level_dir)
    mkpath(joinpath(base_dir, "logs"))
    top_level_pc_fname = joinpath(top_level_dir, "$(task_identifier)_$(fname_idx).jpc")

    println("======= loading wikitext samples =======")
    println("  - data root: $(data_root)")
    trn_data = load_wikitext_data(dataset, "trn", num_tr_samples; data_root = data_root)
    val_data = load_wikitext_data(dataset, "val", num_val_samples; data_root = data_root)
    num_positions = size(trn_data, 2)
    @printf("  - train: %s, val: %s, positions: %d\n", string(size(trn_data)), string(size(val_data)), num_positions)

    println("======= loading low-level progressive-growing PCs =======")
    position_pcs, num_hidden_cats = collect_position_pcs(base_dir, task_identifier, fname_idx, num_positions, num_independent_clusters)
    println("> Top-level hidden categories per position: $(num_hidden_cats) <")

    println("======= generating pseudo datasets for the top-level PC =======")
    patch_level_tr_data = build_top_level_pseudo_data(trn_data, position_pcs, num_hidden_cats; batch_size = batch_size)
    patch_level_val_data = build_top_level_pseudo_data(val_data, position_pcs, num_hidden_cats; batch_size = batch_size)

    clt_edges = chain_clt_edges(num_positions)
    position_idxs_dict = Dict(i => i for i = 1:num_positions)

    get_leaf_pcs(position_idx; num_hidden_cats) = begin
        map(1:num_hidden_cats) do idx
            ps = rand(Float32, num_hidden_cats) .* 0.01
            ps[idx] += 1.0
            ps ./= sum(ps)
            PlainInputNode(position_idxs_dict[position_idx], Categorical(log.(ps)))
        end
    end

    get_edge_params(position_idx1, position_idx2; num_hidden_cats) = begin
        zeros(Float32, num_latents, num_hidden_cats) .+ 0.2
    end

    print("> Constructing top-level PC...")
    t = @elapsed top_level_pc = customized_hclt_p2(clt_edges, num_hidden_cats, num_latents; get_leaf_pcs, get_edge_params, parameterize_leaf_edge = true)
    init_parameters(top_level_pc; perturbation = 0.4)
    @printf(" done (%.2fs)\n", t)

    print("> Moving PC to GPU...")
    t = @elapsed bpc = CuBitsProbCircuit(top_level_pc)
    @printf(" done (%.2fs)\n", t)

    trn_gpu = cu(patch_level_tr_data)
    val_gpu = cu(patch_level_val_data)

    println("> Training top-level PC...")
    mini_batch_em_with_reg(bpc, trn_gpu, num_epochs1;
        batch_size = batch_size, param_inertia = param_inertia1, param_inertia_end = param_inertia2,
        pseudocount = pseudocount, soft_reg = 0.0, soft_reg_width = 3, ent_reg = 0.0,
        log_mode = "plain", verbose = true, eval_dataset = val_gpu, eval_interval = 3)
    mini_batch_em_with_reg(bpc, trn_gpu, num_epochs2;
        batch_size = batch_size, param_inertia = param_inertia2, param_inertia_end = param_inertia3,
        pseudocount = pseudocount, soft_reg = 0.0, soft_reg_width = 5, ent_reg = 0.0,
        log_mode = "plain", verbose = true, eval_dataset = val_gpu, eval_interval = 3)
    update_parameters(bpc)

    tr_ll = loglikelihood_probcat(bpc, trn_gpu; batch_size = batch_size)
    val_ll = loglikelihood_probcat(bpc, val_gpu; batch_size = batch_size)
    tr_bpd = -tr_ll / log(2.0) / num_positions
    val_bpd = -val_ll / log(2.0) / num_positions
    @printf("  - top level model: %2d - train bpd: %.4f - val bpd: %.4f\n", fname_idx, tr_bpd, val_bpd)

    write(top_level_pc_fname, top_level_pc)
    println("> Stored top-level PC at $(top_level_pc_fname)")

    open(ll_file_name, "a") do io
        write(io, @sprintf("\n ====== Wikitext top level bpd: fname %2d - (train %.4f,val %.4f) ====== \n", fname_idx, tr_bpd, val_bpd))
    end
end

function parse_arg(flag, default, T)
    idx = findfirst(==(flag), ARGS)
    idx === nothing && return default
    idx == length(ARGS) && error("Missing value for $(flag)")
    T == String ? ARGS[idx + 1] : parse(T, ARGS[idx + 1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_pg_top_level_wikitext(
        dataset = parse_arg("--dataset", "wikitext", String),
        data_root = parse_arg("--data-root", DEFAULT_DATA_ROOT, String),
        fname_idx = parse_arg("--fname-idx", 4, Int),
        num_independent_clusters = parse_arg("--num-independent-clusters", 200, Int),
        num_init_clusters = parse_arg("--num-init-clusters", 2, Int),
        num_final_clusters = parse_arg("--num-final-clusters", 4, Int),
        num_latents = parse_arg("--num-latents", 64, Int),
        num_tr_samples = parse_arg("--num-tr-samples", 20_000, Int),
        num_val_samples = parse_arg("--num-val-samples", 5_000, Int),
        batch_size = parse_arg("--batch-size", 256, Int),
        num_epochs1 = parse_arg("--num-epochs1", 10, Int),
        num_epochs2 = parse_arg("--num-epochs2", 10, Int),
        gpu = parse_arg("--gpu", 0, Int),
    )
end

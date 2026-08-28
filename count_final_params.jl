using Printf
using ProbabilisticCircuits
import ProbabilisticCircuits: SumEdge, islast, dist, num_parameters

include("VariationalJuice.jl/src/VariationalJuice.jl")
include("VariationalJuice.jl/src-jl/LatentPCs.jl")

# independent=false counts stored scalar log-prob entries.
# independent=true subtracts one normalization constraint per categorical/sum group.
function bit_param_count(bpc; independent::Bool=false, skip_sum_parent_id=nothing)
    edge_params = 0
    edge_groups = 0

    for e in bpc.edge_layers_up.vectors
        if e isa SumEdge
            if skip_sum_parent_id !== nothing && e.parent_id == skip_sum_parent_id
                continue
            end
            edge_params += 1
            if independent && islast(e.tag)
                edge_groups += 1
            end
        end
    end

    input_params = 0
    for node_id in bpc.input_node_ids
        input_params += Int(num_parameters(dist(bpc.nodes[Int(node_id)]), independent))
    end

    edge_params - edge_groups + input_params
end

pc_param_count(pc::ProbCircuit; independent::Bool=false) =
    bit_param_count(BitsProbCircuit(pc); independent=independent)

# Low-level final_pc_*.jpc files are multi-head PCs saved as summate(pcs...).
# Count the children, but exclude the artificial multi-head root parameters.
function mhpc_param_count(pcs::Vector{<:ProbCircuit}; independent::Bool=false)
    fake_pc = summate(pcs...)
    bpc = BitsProbCircuit(fake_pc)
    fake_root_id = UInt32(length(bpc.nodes))
    bit_param_count(bpc; independent=independent, skip_sum_parent_id=fake_root_id)
end

function count_imagenet(image_size::Integer;
        temp_root = joinpath(pwd(), "exps", "progressive_growing", "temp"),
        num_independent_clusters = 400,
        num_init_clusters = 2,
        num_final_clusters = 4,
        fname_idx = 4,
        independent = false)

    base = joinpath(temp_root, "temp_imagenet$(image_size)")
    task = "id_id$(num_independent_clusters)_init$(num_init_clusters)_final$(num_final_clusters)"

    top_file = joinpath(base, "top_level_pcs", "$(task)_$(fname_idx).jpc")
    isfile(top_file) || error("Missing top-level PC: $(top_file)")

    top_params = pc_param_count(read(top_file, ProbCircuit); independent=independent)

    low_params = 0
    low_files = 0
    low_heads = 0

    for cid in 1:num_independent_clusters
        f = joinpath(base, "final_pcs", task, string(cid), "final_pc_$(fname_idx).jpc")
        if isfile(f)
            pcs = read_mhpc(f)
            low_params += mhpc_param_count(pcs; independent=independent)
            low_files += 1
            low_heads += length(pcs)
        end
    end

    total = top_params + low_params

    @printf("\nImageNet%d params (%s):\n", image_size, independent ? "free/independent" : "stored scalars")
    @printf("  top-level:  %d\n", top_params)
    @printf("  low-level:  %d  (%d files, %d heads)\n", low_params, low_files, low_heads)
    @printf("  total:      %d\n", total)

    return (; top_params, low_params, total, low_files, low_heads)
end

function count_wikitext(;
        dataset = "wikitext",
        temp_root = joinpath(pwd(), "exps", "progressive_growing", "temp"),
        num_positions = 32,
        num_independent_clusters = 200,
        num_init_clusters = 2,
        num_final_clusters = 4,
        fname_idx = 4,
        low_level_latents = 2,
        independent = false)

    base = joinpath(temp_root, "temp_$(dataset)")
    task = "id_poswise_cat_l$(low_level_latents)_id$(num_independent_clusters)_init$(num_init_clusters)_final$(num_final_clusters)"

    top_file = joinpath(base, "top_level_pcs", "$(task)_$(fname_idx).jpc")
    isfile(top_file) || error("Missing top-level PC: $(top_file)")

    top_params = pc_param_count(read(top_file, ProbCircuit); independent=independent)

    low_params = 0
    low_files = 0
    low_heads = 0

    for pos in 1:num_positions
        for local_cid in 1:num_independent_clusters
            cluster_id = (pos - 1) * num_independent_clusters + local_cid
            f = joinpath(base, "final_pcs", task, string(cluster_id), "final_pc_$(fname_idx).jpc")
            if isfile(f)
                pcs = read_mhpc(f)
                low_params += mhpc_param_count(pcs; independent=independent)
                low_files += 1
                low_heads += length(pcs)
            end
        end
    end

    total = top_params + low_params

    @printf("\nWikiText params (%s):\n", independent ? "free/independent" : "stored scalars")
    @printf("  top-level:  %d\n", top_params)
    @printf("  low-level:  %d  (%d files, %d heads)\n", low_params, low_files, low_heads)
    @printf("  total:      %d\n", total)

    return (; top_params, low_params, total, low_files, low_heads)
end

# Examples:
count_imagenet(32)
# count_imagenet(64)
count_wikitext(num_positions=32)

# If you want normalized degrees of freedom instead of stored scalar entries:
# count_imagenet(32; independent=true)
# count_wikitext(num_positions=32; independent=true)


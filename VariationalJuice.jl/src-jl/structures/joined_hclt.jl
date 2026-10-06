using Base: AbstractArrayOrBroadcasted
using MetaGraphs: MetaDiGraph, outneighbors, get_prop, set_prop!
using CUDA
using ProbabilisticCircuits
using ChowLiuTrees: learn_chow_liu_tree, topk_MST
using Printf

import Base: hash # extend

function myhash(rnode::PartitionNode)
    ch_scopes = map(rn -> rn.scope, rnode.children)
    first_ids = map(s -> first(s), ch_scopes)
    ch_scopes = ch_scopes[sortperm(first_ids)]
    hash(ch_scopes)
end

function clts2rgraph(clts::Vector{MetaDiGraph}, num_vars::Integer)
    scope2rnode = Dict{BitSet,RegionGraph}()
    for clt in clts
        var_seq = PCs.bottom_up_order(clt)
        for curr_var in var_seq
            ch_vars = outneighbors(clt, curr_var)

            rnode = if length(ch_vars) == 0
                scope = BitSet([curr_var])
                if !(scope in keys(scope2rnode))
                    scope2rnode[scope] = InputRegionNode(0, scope)
                end
                scope2rnode[scope]
            else
                scope = mapreduce(ch_var -> get_prop(clt, ch_var, :rnode).scope, union, ch_vars)
                scope = union(scope, BitSet([curr_var]))
                ch_rnodes = Vector{RegionGraph}(undef, length(ch_vars))
                map!(ch_var -> get_prop(clt, ch_var, :rnode), ch_rnodes, ch_vars)
                in_rnode = get!(scope2rnode, BitSet([curr_var])) do
                    InputRegionNode(0, BitSet([curr_var]))
                end
                push!(ch_rnodes, in_rnode)

                if scope in keys(scope2rnode)
                    inner_rn = scope2rnode[scope]
                    redundant = false
                    p_rnode = PartitionNode(0, ch_rnodes)
                    for i = 1 : length(inner_rn.children)
                        if myhash(inner_rn.children[i]) == myhash(p_rnode)
                            redundant = true
                            break
                        end
                    end
                    if !redundant
                        push!(inner_rn.children, p_rnode)
                    end
                    inner_rn
                else
                    rnode = InnerRegionNode(0, [PartitionNode(0, ch_rnodes)])
                    scope2rnode[scope] = rnode
                    rnode
                end
            end
            set_prop!(clt, curr_var, :rnode, rnode)
        end
    end
    scope2rnode[BitSet(collect(1:num_vars))]
end

function joined_hclt(token_cid_datasets::Vector, token_feature_datasets::Vector, num_hidden_cats;  num_cats = nothing, shape = :directed,
                     input_type = Literal, pseudocount = 0.1)

    @assert length(token_cid_datasets) == length(token_feature_datasets)
    num_vars = size(token_feature_datasets[1], 2)
    @assert size(token_cid_datasets[1], 2) == num_vars

    # Get all CLTs
    println("> Constructing CLTs...")
    clts = Vector{MetaDiGraph}()
    for (i, data) in enumerate(token_feature_datasets)
        print(@sprintf("  - CLT #%03d/%03d... ", i, length(token_feature_datasets)))
        t = @elapsed begin
            # if data isa Array
            #     data = cu(data)
            # end
            clt_edges = learn_chow_liu_tree_from_features(data; pseudocount = pseudocount, Float = Float32)
            clt = PCs.clt_edges2graphs(clt_edges; shape)
            push!(clts, clt)
        end
        println(@sprintf("done (%.2fs)", t))
    end

    # Construct region graph
    rnode = clts2rgraph(clts, num_vars)

    # Region graph
    f_input(rn)::Vector{<:ProbCircuit} = begin
        if input_type == Categorical
            [PlainInputNode(randvar(rn), Categorical(num_cats)) for _ = 1 : num_hidden_cats]
        else
            error("Unknown input type $(input_type).")
        end
    end
    f_partition(rn, ins)::Vector{<:ProbCircuit} = begin
        [multiply([cs[i] for cs in ins]...) for i = 1 : num_hidden_cats]
    end
    f_inner(rn, ins)::Vector{<:ProbCircuit} = begin
        flattened_ins = reduce(vcat, ins)
        [summate(flattened_ins...) for _ = 1 : num_hidden_cats]
    end
    foldup_aggregate(rnode, f_input, f_partition, f_inner, Vector{<:ProbCircuit})
end

function learn_chow_liu_tree_from_features(features;
        num_trees=1, dropout_prob=0.0, weights=nothing,
        pseudocount=1.0, Float=Float32)
    distances = pairwise_distances(features; weights, pseudocount, Float)
    if distances isa CuArray
        distances_cpu = Array(distances)
        CUDA.unsafe_free!(distances)
    else
        distances_cpu = distances
    end

    trees = topk_MST(-distances_cpu; num_trees, dropout_prob = Float64(dropout_prob))
    num_trees == 1 ? trees[1] : trees
end

function pairwise_distances(data::AbstractArray{<:Real,3};
                     weights::Union{Vector, Nothing} = nothing,
                     pseudocount = 1f-6,
                     Float = Float32)
    num_samples = size(data, 1)
    num_vars = size(data, 2)
    embed_size = size(data, 3)

    # Treat each random variable as the concatenation of its embedding vectors
    # across samples, and use the (weighted) cosine similarity between those
    # vectors.  This yields one floating-point similarity for every pair of
    # random variables while preserving the input's device (CPU or GPU).
    x = permutedims(data, (1, 3, 2))
    x = reshape(x, num_samples * embed_size, num_vars)
    x = Float.(x)

    if weights !== nothing
        sample_weights = weights
        if data isa CuArray
            sample_weights = CuArray(sample_weights)
        end
        sample_weights = sqrt.(Float.(sample_weights))
        sample_weights = repeat(sample_weights, outer = size(data, 3))
        x .*= reshape(sample_weights, :, 1)
    end

    norms = sqrt.(sum(abs2, x; dims = 1) .+ Float(pseudocount))
    sims = (transpose(x) * x) ./ (transpose(norms) * norms)
    sims
end

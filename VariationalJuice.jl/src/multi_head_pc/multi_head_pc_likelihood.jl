
# NOTE (packaging): `extract_lls_from_root_nodes` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).




function eval_multi_head_pc(mars, mhbpc, data, head_mask, example_ids; mine, maxe, soft_reg, soft_reg_width, debug = false)
    sum_agg_func(x::Float32, y::Float32) = 
        logsumexp(x, y)

    bpc = mhbpc.bpc

    init_mar!_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        PCs.layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function multihead_loglikelihoods_probcat(mhbpc::CuMultiHeadBitsProbCircuit, data::CuArray, head_mask::CuMatrix; batch_size, mars_mem = nothing, 
                                          mine = 2, maxe = 32)
    bpc = mhbpc.bpc
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    lls = prep_memory(nothing, (batch_size,), (false,))

    log_likelihoods = CUDA.zeros(Float32, num_examples)

    for batch_start = 1 : batch_size : num_examples

        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        eval_multi_head_pc(mars, mhbpc, data, head_mask, batch; mine, maxe, soft_reg = 0.0, soft_reg_width = 1)
        lls = extract_lls_from_root_nodes(mars, mhbpc, head_mask, batch, lls)

        log_likelihoods[batch_start:batch_end] .= lls[1:num_batch_examples]
    end

    PCs.cleanup_memory(mars, mars_mem)

    log_likelihoods
end

function multihead_loglikelihood_probcat(mhbpc::CuMultiHeadBitsProbCircuit, data::CuArray, head_mask::CuMatrix; batch_size, mars_mem = nothing, 
                                         mine = 2, maxe = 32)
    lls = multihead_loglikelihoods_probcat(mhbpc, data, head_mask; batch_size, mars_mem, mine, maxe)

    return sum(lls) / length(lls)
end
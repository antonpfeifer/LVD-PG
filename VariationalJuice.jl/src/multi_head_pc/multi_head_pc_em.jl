using Random


function large_dataset_training(
        mhbpc::CuMultiHeadBitsProbCircuit, data::Array, head_mask::Array, num_epochs; 
        num_samples_per_epoch, batch_size, pseudocount, soft_reg, soft_reg_width,
        param_inertia, param_inertia_end, num_clusters = nothing
    )

    bpc = mhbpc.bpc

    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    mars_mem = prep_memory(nothing, (batch_size, num_nodes), (false, true))
    flows_mem = prep_memory(nothing, (batch_size, num_nodes), (false, true))
    node_aggr_mem = prep_memory(nothing, (num_nodes,))
    edge_aggr_mem = prep_memory(nothing, (num_edges,))
    lls_mem = prep_memory(nothing, (batch_size,), (false,))

    Δparam_inertia = (param_inertia_end-param_inertia) / num_epochs

    for epoch = 1 : num_epochs
        sample_ids = randperm(num_examples)[1:num_samples_per_epoch]
        epoch_data = data[sample_ids, :]
        if ndims(head_mask) == 1
            epoch_head_mask = head_mask[sample_ids]
            epoch_head_mask = Float32.(collect(0:num_clusters-1)' .== epoch_head_mask) # one-hot encoding
        elseif ndims(head_mask) == 2
            epoch_head_mask = head_mask[sample_ids, :]
        else
            error("Not implemented")
        end

        mini_batch_em_for_multihead_pc(mhbpc, cu(epoch_data), cu(epoch_head_mask), 1; batch_size, pseudocount, 
                                       soft_reg, soft_reg_width, param_inertia, mars_mem, flows_mem, node_aggr_mem,
                                       edge_aggr_mem, init_clear_mem = (epoch == 1))
    end

    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))

    nothing
end

# NOTE (packaging): `mini_batch_em_for_multihead_pc` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).
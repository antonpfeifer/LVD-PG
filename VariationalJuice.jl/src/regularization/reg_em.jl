using Random
using StatsFuns
using DSP


# NOTE (packaging): `apply_entropy_reg_kernel` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `apply_entropy_reg` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `update_params_with_reg_kernel` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

# NOTE (packaging): `update_params_with_reg` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `weighted_ll` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `overprint` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

function mini_batch_em_with_reg(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; batch_size, pseudocount, soft_reg, soft_reg_width, ent_reg,
                                param_inertia, param_inertia_end = param_inertia, flow_memory = zero(Float32), shuffle = :each_epoch,
                                mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                                log_params_mem = nothing, td_probs_mem = nothing, node_cum1_mem = nothing, node_cum2_mem = nothing, 
                                sib_count_mem = nothing, mine = 2, maxe = 32, debug = false, verbose = true, weights = nothing,
                                eval_dataset = nothing, eval_weights = nothing, eval_interval = 0, log_mode = "overprint")
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples
    @assert isnothing(weights) || length(weights) == num_examples

    if verbose && log_mode == "overprint"
        println("Preparing to run mini-batch EM...")
    end

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))

    log_params = prep_memory(log_params_mem, (num_edges,))
    td_probs = prep_memory(td_probs_mem, (num_nodes,))
    node_cum1 = prep_memory(node_cum1_mem, (num_nodes,))
    node_cum2 = prep_memory(node_cum2_mem, (num_nodes,))
    sib_count = prep_memory(sib_count_mem, (num_nodes,))

    up2downedge = zeros(UInt32, num_edges)
    for (down_edge_idx, up_edge_idx) in enumerate(Array(bpc.down2upedge))
        up2downedge[up_edge_idx] = down_edge_idx
    end
    up2downedge = cu(up2downedge)

    edge_aggr .= zero(Float32)
    PCs.clear_input_node_mem(bpc; rate = 0, debug)

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()

    Δparam_inertia = (param_inertia_end-param_inertia)/num_epochs

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)

    last_test_ll = nothing

    for epoch = 1 : num_epochs

        log_likelihoods_epoch .= zero(Float32)
        CUDA.synchronize()
        
        (shuffle == :each_epoch) && do_shuffle()
        CUDA.synchronize()
        
        for (batch_id, batch) in enumerate(batches)
            (shuffle == :each_batch) && do_shuffle()

            if iszero(flow_memory)
                edge_aggr .= zero(Float32)
                PCs.clear_input_node_mem(bpc; rate = 0, debug)
            else
                # slowly forget old edge aggregates
                rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                edge_aggr .*= rate
                PCs.clear_input_node_mem(bpc; rate)
            end
            
            probs_flows_circuit_with_reg(flows, mars, edge_aggr, bpc, data, batch; 
                                         mine, maxe, soft_reg, soft_reg_width, weights)
            CUDA.synchronize()
            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1:1], mars[1:batch_size,end:end])
            
            # to modify
            PCs.add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount; debug)
            PCs.aggr_node_flows(node_aggr, bpc, edge_aggr; debug)
            # apply_entropy_reg(edge_aggr, node_aggr, bpc, log_params, td_probs, node_cum1, 
            #                   node_cum2, sib_count, ent_reg, up2downedge; debug)
            # update_params_with_reg(bpc, log_params; inertia = param_inertia, debug)
            PCs.update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia, debug)
            
            PCs.update_input_node_params(bpc; pseudocount, inertia = param_inertia, debug)

        end
        CUDA.synchronize()
        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)
        CUDA.synchronize()

        param_inertia += Δparam_inertia

        if verbose
            if eval_dataset !== nothing && eval_interval > 0 && epoch % eval_interval == 0
                test_ll = loglikelihood_probcat(bpc, eval_dataset, eval_weights; batch_size)
                if log_mode == "overprint"
                    overprint("Mini-batch EM epoch $epoch/$num_epochs: train LL $log_likelihood - test LL $test_ll")
                else
                    println("Mini-batch EM epoch $epoch/$num_epochs: train LL $log_likelihood - test LL $test_ll")
                end
                last_test_ll = test_ll
            else
                if log_mode == "overprint"
                    if last_test_ll !== nothing
                        overprint("Mini-batch EM epoch $epoch/$num_epochs; train LL $log_likelihood - test LL $last_test_ll")
                    else
                        overprint("Mini-batch EM epoch $epoch/$num_epochs; train LL $log_likelihood")
                    end
                else
                    println("Mini-batch EM epoch $epoch/$num_epochs; train LL $log_likelihood")
                end
            end
        end
    end

    PCs.cleanup_memory((flows, flows_mem), (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    log_likelihoods
end

# NOTE (packaging): `init_parameters_by_logits` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `init_parameters_by_logits` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `init_parameters_by_logits` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `perturb_parameters` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `perturb_parameters` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

# NOTE (packaging): `perturb_parameters` removed here — src-jl/ copy wins (see src/VariationalJuice.jl).

function reg_mini_batch_em(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; 
                           batch_size, update_interval, pseudocount, 
                           param_inertia, param_inertia_end = param_inertia, 
                           flow_memory = 0, flow_memory_end = flow_memory, 
                           shuffle=:each_epoch,  
                           mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                           mine = 2, maxe = 32, debug = false, verbose = true,
                           callbacks = [], eval_dataset = nothing, eval_interval = 0)

    @assert pseudocount >= 0
    @assert 0 <= param_inertia <= 1
    @assert param_inertia <= param_inertia_end <= 1
    @assert 0 <= flow_memory  
    @assert flow_memory <= flow_memory_end  
    @assert shuffle ∈ [:once, :each_epoch, :each_batch]
    
    insert!(callbacks, 1, PCs.MiniBatchLog(verbose))
    callbacks = PCs.CALLBACKList(callbacks)
    PCs.init(callbacks; batch_size, bpc)

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))

    edge_aggr .= zero(Float32)
    PCs.clear_input_node_mem(bpc; rate = 0, debug)

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()

    Δparam_inertia = (param_inertia_end-param_inertia)/num_epochs
    Δflow_memory = (flow_memory_end-flow_memory)/num_epochs

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)

    for epoch in 1:num_epochs

        log_likelihoods_epoch .= zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        edge_aggr .= zero(Float32)
        PCs.clear_input_node_mem(bpc; rate = 0, debug)

        for (batch_id, batch) in enumerate(batches)

            (shuffle == :each_batch) && do_shuffle()
            
            if batch_id % update_interval == 1
                if iszero(flow_memory)
                    edge_aggr .= zero(Float32)
                    PCs.clear_input_node_mem(bpc; rate = 0, debug)
                else
                    # slowly forget old edge aggregates
                    rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                    edge_aggr .*= rate
                    PCs.clear_input_node_mem(bpc; rate)
                end
            end

            PCs.probs_flows_circuit(flows, marginals, edge_aggr, bpc, data, batch; 
                                mine, maxe, debug)
            
            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1:1],
                    marginals[1:batch_size,end:end])

            if batch_id % update_interval == 0
                PCs.add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount; debug)
                PCs.aggr_node_flows(node_aggr, bpc, edge_aggr; debug)
                PCs.update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia, debug)
                
                PCs.update_input_node_params(bpc; pseudocount, inertia = param_inertia, debug)
            end
            
        end
        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)
        PCs.call(callbacks, epoch, log_likelihood)

        param_inertia += Δparam_inertia
        flow_memory += Δflow_memory
    end

    PCs.cleanup_memory((flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    PCs.cleanup(callbacks)

    log_likelihoods
end
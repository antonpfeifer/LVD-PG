

# NOTE (packaging): `select_gpu` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

function num_categories(mbpc::CuMetaBitsProbCircuit)
    nodes = Array(mbpc.bpc.nodes)
    num_cats = 0
    for n in nodes
        if n isa BitsInput
            ncats = dist(n).num_cats
            if ncats > num_cats
                num_cats = ncats
            end
        end
    end
    num_cats
end
function num_categories(pc::ProbCircuit)
    num_cats = 0
    foreach(pc) do n
        if n isa PlainInputNode
            ncats = num_categories(dist(n))
            if ncats > num_cats
                num_cats = ncats
            end
        end
    end
    num_cats
end

function leaf_params(mbpc::CuMetaBitsProbCircuit, num_vars, n_hiddens; num_cats = 256)
    pc_cat_params = zeros(Float32, num_vars, n_hiddens, num_cats)
    node_idx = ones(Int32, num_vars)
    nodes = Array(mbpc.bpc.nodes)
    heap = Array(mbpc.bpc.heap)
    for node in nodes
        if node isa BitsInput
            v = node.variable
            heap_start = node.dist.heap_start
            pc_cat_params[v, node_idx[v], :] .= heap[heap_start:heap_start+num_cats-1]
            node_idx[v] += 1
        end
    end
    pc_cat_params
end

# NOTE (packaging): `get_randvars` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

# NOTE (packaging): `issmooth` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

# NOTE (packaging): `isdecomposable` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).

# NOTE (packaging): `isvalid` removed here — identical copy kept in src-jl/ (see src/VariationalJuice.jl).
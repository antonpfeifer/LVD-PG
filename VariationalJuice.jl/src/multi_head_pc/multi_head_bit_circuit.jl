using ProbabilisticCircuits: BitsNode, tag_at


struct CuMultiHeadBitsProbCircuit{BitsNodes <: BitsNode}

    # the original BitPC
    bpc::CuBitsProbCircuit{BitsNodes}

    # ids of the root nodes
    root_ids::CuVector{UInt32}

end

# NOTE (packaging): outer constructor removed here — src-jl/ copy wins (see src/VariationalJuice.jl).
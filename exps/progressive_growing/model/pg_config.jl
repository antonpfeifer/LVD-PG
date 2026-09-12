struct ProgressiveGrowingConfig
    num_token_clusters::Int
    num_sentence_clusters::Int
    num_hclt_latents::Int
    num_init_clusters::Int
    num_final_clusters::Int
    batch_size::Int
    max_grow_frac::Float64
    prune_threshold::Float32
    token_emission_alpha::Float32
end

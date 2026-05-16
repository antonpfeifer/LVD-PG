function sample_cat(weights)
    r = rand() * sum(weights)
    for (i, w) in enumerate(weights)
        r -= w
        if r <= 0
            return i
        end
    end
    return length(weights)
end

function ancestral_sample(node::ProbabilisticCircuits.PlainSumNode)
    weights = exp.(node.params)
    idx = sample_cat(weights)
    return ancestral_sample(node.inputs[idx])
end

function ancestral_sample(node::ProbabilisticCircuits.PlainMulNode)
    res = Dict{Int, Int}()
    for c in node.inputs
        merge!(res, ancestral_sample(c))
    end
    return res
end

function ancestral_sample(node::ProbabilisticCircuits.PlainInputNode)
    ps = exp.(node.dist.logps)
    val = sample_cat(ps)
    return Dict(first(node.randvars) => val)
end

function ycrcb2rgb(imgs)
    new_imgs = zeros(UInt8, size(imgs)...)
    new_imgs[:,1,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* 0 .+ (imgs[:,3,:,:] .- 128) .* 1.5748, 0, 255)))
    new_imgs[:,2,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* -0.1873 .+ (imgs[:,3,:,:] .- 128) .* -0.4681, 0, 255)))
    new_imgs[:,3,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* 1.8556 .+ (imgs[:,3,:,:] .- 128) .* 0, 0, 255)))
    return new_imgs
end

image_size = 32
fname_idx = 4
num_independent_clusters = 400
num_init_clusters = 2
num_final_clusters = 4
task_identifier = "id_id400_init2_final4"

base_dir = "exps/progressive_growing/temp/temp_imagenet$(image_size)"
top_level_pc_fname = joinpath(base_dir,"top_level_pcs/$(task_identifier)_$(fname_idx).jpc")

top_level_pc = read(top_level_pc_fname, ProbCircuit)

valid_clusters = []
n_clusters_y = 0
for cluster_id = 1 : num_independent_clusters
    base_dir1 = joinpath(base_dir,"final_pcs/$(task_identifier)/$(cluster_id)")
    final_pc_fname = joinpath(base_dir1, "final_pc_$(fname_idx).jpc")
    if !isfile(final_pc_fname)
        continue
    end
    pc = read_mhpc(final_pc_fname)
    push!(valid_clusters, (cluster_id, length(pc), final_pc_fname, pc))
    global n_clusters_y += length(pc)
end

function map_cluster_to_low_level_pc(cid)
    current_y_offset = 1
    for (cluster_id, num_clusters, final_pc_fname, pc) in valid_clusters
        if cid >= current_y_offset && cid < current_y_offset + num_clusters
            root_idx = cid - current_y_offset + 1
            return pc[root_idx]
        end
        current_y_offset += num_clusters
    end
    error("Cluster ID out of bounds")
end

function generate_image()
    # 1. Sample top level PC
    top_sample = ancestral_sample(top_level_pc)
    
    # Image will be 3 x 32 x 32
    patch_size = 8
    patches_per_row = 4
    
    img_ycrcb = zeros(Float32, 1, 3, 32, 32)
    
    for (patch_idx_1based, cluster_id) in top_sample
        # patch_idx_1based is from 1 to 16
        # Need to map patch_idx to coordinates. In LatentPCs.jl/train_PG_top_level_pcs.jl:
        # patch_idx is from train_PG_top_level_pcs, typically 1 to 16 mapped arbitrarily?
        # Wait, in the output earlier we saw:
        # Patch indices: [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
        # and they were mapped to 1..16.
        # But wait, we can just reshape linearly.
        patch_id_0based = patch_idx_1based - 1
        px = patch_id_0based % patches_per_row
        py = patch_id_0based ÷ patches_per_row
        
        # Sample from the low level PC
        low_pc = map_cluster_to_low_level_pc(cluster_id)
        low_sample = ancestral_sample(low_pc)
        
        # Fill the patch. The variables in low_sample are 1..192
        # Data format was reshape(subsampled_tr_data[:,:,x_s:x_e,y_s:y_e],(:,3*(patch_size^2)))
        # So variables 1..192 map to (3, 8, 8)
        patch_pixels = zeros(Float32, 192)
        for k in 1:192
            # 1-indexed categorical values 1..256 -> 0..255
            patch_pixels[k] = get(low_sample, k, 1) - 1
        end
        patch_pixels = reshape(patch_pixels, 3, 8, 8)
        
        x_s = py * patch_size + 1
        x_e = (py + 1) * patch_size
        y_s = px * patch_size + 1
        y_e = (px + 1) * patch_size
        
        img_ycrcb[1, :, x_s:x_e, y_s:y_e] = patch_pixels
    end
    
    img_rgb = ycrcb2rgb(img_ycrcb)
    return img_rgb
end
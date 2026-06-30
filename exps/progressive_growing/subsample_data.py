"""
Subsample the wikitext training data to fit in RAM.
Creates new files with a random subset of samples.
"""
import numpy as np
import os

DATA_DIR = "data/data_wikitext"
N_SAMPLES = 2_000_000  # Target number of training samples (adjust as needed)

# Load with mmap to avoid reading everything into RAM
print("Opening memmapped files...")
data_trn = np.load(os.path.join(DATA_DIR, "data_trn.npy"), mmap_mode="r")
feat_trn = np.load(os.path.join(DATA_DIR, "idfeat_trn.npy"), mmap_mode="r")

total = data_trn.shape[0]
print(f"Total training samples: {total}")
print(f"Subsampling to: {N_SAMPLES}")

if N_SAMPLES >= total:
    print("N_SAMPLES >= total, nothing to do.")
    exit(0)

# Random subset indices (sorted for sequential disk access on mmap)
rng = np.random.default_rng(42)
indices = np.sort(rng.choice(total, size=N_SAMPLES, replace=False))

# Create output arrays as memmaps
BACKUP_SUFFIX = "_full"
out_data_path = os.path.join(DATA_DIR, "data_trn.npy")
out_feat_path = os.path.join(DATA_DIR, "idfeat_trn.npy")

# Rename originals
for path in [out_data_path, out_feat_path]:
    backup = path.replace(".npy", f"{BACKUP_SUFFIX}.npy")
    if not os.path.exists(backup):
        print(f"Renaming {path} -> {backup}")
        os.rename(path, backup)
    else:
        print(f"Backup already exists: {backup}")

# Re-open from backups
data_trn = np.load(out_data_path.replace(".npy", f"{BACKUP_SUFFIX}.npy"), mmap_mode="r")
feat_trn = np.load(out_feat_path.replace(".npy", f"{BACKUP_SUFFIX}.npy"), mmap_mode="r")

# Write subsampled data
print("Writing subsampled data_trn...")
out_data = np.lib.format.open_memmap(
    out_data_path, mode="w+", dtype=data_trn.dtype, shape=(N_SAMPLES, data_trn.shape[1])
)
CHUNK = 10000
for start in range(0, N_SAMPLES, CHUNK):
    end = min(start + CHUNK, N_SAMPLES)
    out_data[start:end] = data_trn[indices[start:end]]
    if start % 100000 == 0:
        print(f"  data: {start}/{N_SAMPLES}")
out_data.flush()
del out_data

print("Writing subsampled idfeat_trn...")
out_feat = np.lib.format.open_memmap(
    out_feat_path, mode="w+", dtype=feat_trn.dtype, shape=(N_SAMPLES, feat_trn.shape[1], feat_trn.shape[2])
)
for start in range(0, N_SAMPLES, CHUNK):
    end = min(start + CHUNK, N_SAMPLES)
    out_feat[start:end] = feat_trn[indices[start:end]]
    if start % 100000 == 0:
        print(f"  feat: {start}/{N_SAMPLES}")
out_feat.flush()
del out_feat

print("Done! Subsampled files written.")
print(f"  {out_data_path}: ({N_SAMPLES}, {data_trn.shape[1]})")
print(f"  {out_feat_path}: ({N_SAMPLES}, {feat_trn.shape[1]}, {feat_trn.shape[2]})")

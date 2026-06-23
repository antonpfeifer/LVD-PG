import os

import faiss
import numpy as np


def _train_2d_kmeans(train_features, n_clusters, gpu_id=0, centroids=None):
    train_features = np.ascontiguousarray(train_features.astype(np.float32))
    kmeans = faiss.Clustering(train_features.shape[1], n_clusters)
    if centroids is not None:
        faiss.copy_array_to_vector(
            np.ascontiguousarray(np.array(centroids).astype(np.float32).reshape(-1)),
            kmeans.centroids,
        )
    kmeans.verbose = False
    kmeans.niter = 200
    kmeans.nredo = 5

    try:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id
        index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(), train_features.shape[1], cfg
        )
    except Exception as e:
        print(f"GPU Index failed: {e}. Falling back to CPU.")
        index = faiss.IndexFlatL2(train_features.shape[1])

    kmeans.train(train_features, index)
    return faiss.vector_float_to_array(kmeans.centroids).reshape(
        n_clusters, train_features.shape[1]
    )


def train_kmeans_model(train_features, n_clusters, gpu_id=0, centroids=None):
    print("Num GPUs detected by faiss:", faiss.get_num_gpus())
    print("Features shape:", train_features.shape)

    # Important: do not call np.asarray(train_features) before checking ndim.
    # For mmap'ed 3D wikitext features this would materialize the full ~147GB file.
    ndim = getattr(train_features, "ndim", None)
    if ndim is None:
        train_features = np.asarray(train_features)
        ndim = train_features.ndim

    if ndim == 3:
        position_centroids = []
        cents = None if centroids is None else np.asarray(centroids)
        for pos in range(train_features.shape[1]):
            pos_centroids = None if cents is None else cents[pos]
            position_centroids.append(
                _train_2d_kmeans(train_features[:, pos, :], n_clusters, gpu_id, pos_centroids)
            )
        return np.stack(position_centroids, axis=0)

    return _train_2d_kmeans(train_features, n_clusters, gpu_id, centroids)


def pred_kmeans_clusters(centroids, features):
    centroids = np.asarray(centroids).astype(np.float32, copy=False)

    # Important: handle 3D mmap'ed features position by position. Calling
    # np.asarray(features).astype(...) here would materialize the full file.
    ndim = getattr(features, "ndim", None)
    if ndim is None:
        features = np.asarray(features)
        ndim = features.ndim

    if ndim == 3:
        labels_by_position = np.empty(features.shape[:2], dtype=np.int64)
        n_clusters = centroids.shape[1]
        for pos in range(features.shape[1]):
            pos_features = np.ascontiguousarray(features[:, pos, :].astype(np.float32, copy=False))
            pos_centroids = np.ascontiguousarray(centroids[pos])
            index = faiss.IndexFlatL2(pos_centroids.shape[1])
            index.add(pos_centroids)
            _, labels = index.search(pos_features, 1)
            labels_by_position[:, pos] = labels.ravel() + pos * n_clusters
        return labels_by_position.ravel() + 1

    features = np.ascontiguousarray(np.asarray(features).astype(np.float32, copy=False))
    centroids = np.ascontiguousarray(centroids)
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, labels = index.search(features, 1)
    labels = labels.ravel()

    return labels + 1


def save_kmeans_model(centroids, model_path):
    file_name = os.path.join(model_path, "centroids.npz")
    np.savez(file_name, centroids=centroids)
    np.savez(file_name, centroids=centroids)
    np.savez(file_name, centroids=centroids)

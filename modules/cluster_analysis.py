from typing import Any

import numpy as np
from sklearn.cluster import KMeans


def find_cluster_centroids(embeddings, max_k=10) -> Any:
    if len(embeddings) <= 1:
        return embeddings

    fitted = [KMeans(n_clusters=k, random_state=0).fit(embeddings) for k in range(1, max_k + 1)]
    inertia = [km.inertia_ for km in fitted]
    cluster_centroids = [km.cluster_centers_ for km in fitted]

    diffs = [inertia[i] - inertia[i + 1] for i in range(len(inertia) - 1)]
    if not diffs:
        return cluster_centroids[0]
    return cluster_centroids[diffs.index(max(diffs)) + 1]


def find_closest_centroid(centroids: list, normed_face_embedding) -> tuple:
    centroids = np.array(centroids)
    normed_face_embedding = np.array(normed_face_embedding)
    similarities = np.dot(centroids, normed_face_embedding)
    closest_centroid_index = np.argmax(similarities)
    return closest_centroid_index, centroids[closest_centroid_index]

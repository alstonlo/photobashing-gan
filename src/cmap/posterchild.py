import numpy as np
import numpy.linalg as LA
from sklearn.cluster import KMeans

from src.cmap.simplify_convexhull import ConvexHull, get_faces_vertices, simplified_convex_hull


def posterchild_palettize(image, n_colors, seed):
    flat_image = image.reshape((-1, 3))

    # k-means
    k_means = KMeans(n_clusters=20, random_state=seed).fit(flat_image)
    labels = k_means.labels_.reshape(image.shape[:2])
    clusters = k_means.cluster_centers_
    flat_image = clusters[labels.flatten()]

    assert LA.matrix_rank(flat_image) > 1

    og_hull = ConvexHull(flat_image)
    hvertices, hfaces = get_faces_vertices(og_hull)
    mesh = simplified_convex_hull(n_colors, hvertices, hfaces).vs

    palette = mesh.astype(np.float32)
    dist_matrix = LA.norm(flat_image[:, None, :] - palette[None, :, :], axis=-1)
    labels = dist_matrix.argmin(axis=-1)

    cmap = labels.reshape(image.shape[:2])
    return cmap, palette



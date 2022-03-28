import collections
import itertools

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn_extra.robust import RobustWeightedKMeans

from src.cmap.posterchild import posterchild_palettize


def quantize_colors(
        image, method, blur_sigma=1, n_colors=5,
        simplify=True, blob_threshold=700, min_islands=2, seed=420
):
    assert image.dtype == np.float32
    assert method in {"k_means", "k_means_robust", "posterchild"}
    image = cv2.GaussianBlur(image, (5, 5), blur_sigma)

    # k-means clustering
    if method[0] == "k":
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)  # RGB -> L*a*b*
        flat_lab_image = lab_image.reshape(-1, 3)

        if method[2:] == "means":
            results = KMeans(
                n_clusters=n_colors,
                random_state=seed
            ).fit(flat_lab_image)

        elif method[2:] == "means_robust":
            best_score = float("inf")
            best_results = None

            for _ in range(10):
                km = RobustWeightedKMeans(
                    n_clusters=n_colors,
                    weighting="mom",
                    max_iter=1000,
                    k=10,
                    random_state=seed
                )
                results = km.fit(flat_lab_image)
                cmap = results.cluster_centers_[km.predict(flat_lab_image)]
                score = np.mean((cmap - flat_lab_image) ** 2)
                if score < best_score:
                    best_results, best_score = results, score

            results = best_results

        else:
            raise ValueError()

        cmap = results.labels_.reshape(image.shape[:2])
        lab_palette = results.cluster_centers_  # L*a*b* -> RGB
        palette = cv2.cvtColor(lab_palette[None, :, :], cv2.COLOR_Lab2RGB)[0]

    # posterchild algorithm
    elif method == "posterchild":
        cmap, palette = posterchild_palettize(image, n_colors=n_colors, seed=seed)
        lab_palette = cv2.cvtColor(palette[None, :, :], cv2.COLOR_RGB2LAB)[0]

    else:
        raise ValueError()

    if not simplify:
        image = palette[cmap.flatten()].reshape(image.shape)
        return image

    # create set (archipelago) of color shapes (islands)
    archipelago = []
    chart = dict()
    color_counts = [0] * n_colors

    for root in itertools.product(range(cmap.shape[0]), repeat=2):
        if root in chart:
            continue
        island = _bfs_grow_island(root, cmap)
        archipelago.append(island)
        for idx in island:
            chart[idx] = island
        color_counts[cmap[root]] += 1

    # remove small islands
    lab_palette_pdists = squareform(pdist(lab_palette))
    archipelago.sort(key=lambda x: len(x))

    while True:  # flip colors of islands

        for island in archipelago:
            color = cmap[next(iter(island))]
            if color_counts[color] > min_islands:
                break
        else:
            break

        if len(island) > blob_threshold:
            break

        surroundings = [list() for _ in range(n_colors)]
        for idx in island:
            for nb in _neighbor_indices(idx, cmap.shape[0]):
                if nb is None:
                    continue
                if cmap[idx] != cmap[nb]:
                    s = surroundings[cmap[nb]]
                    if chart[nb] not in s:
                        s.append(chart[nb])

        # recolor by closest color
        new_color = np.argmin([
            (lab_palette_pdists[color][i] if nbs else np.inf)
            for i, nbs in enumerate(surroundings)
        ])

        new_parent = surroundings[new_color][0]
        to_merge = surroundings[new_color][1:]
        to_merge.append(island)

        for idx in island:
            cmap[idx] = new_color
        for child in to_merge:
            for idx in child:
                chart[idx] = new_parent
            archipelago.remove(child)
            new_parent.update(child)

        # bubble new_parent up so archipelago remains sorted
        idx = archipelago.index(new_parent)
        while idx + 2 < len(archipelago) and (len(new_parent) > len(archipelago[idx + 1])):
            archipelago[idx], archipelago[idx + 1] = archipelago[idx + 1], archipelago[idx]
            idx += 1

    image = palette[cmap.flatten()].reshape(image.shape)
    return image


def _neighbor_indices(idx, width):
    i, j = idx
    nbd = []
    if i > 0:
        nbd.append((i - 1, j))
    if j > 0:
        nbd.append((i, j - 1))
    if i + 1 < width:
        nbd.append((i + 1, j))
    if j + 1 < width:
        nbd.append((i, j + 1))
    return nbd


def _bfs_grow_island(root, cmap):
    island = {root}
    queue = collections.deque([root])
    while queue:
        idx = queue.popleft()
        for nb in _neighbor_indices(idx, cmap.shape[0]):
            if (nb not in island) and (cmap[idx] == cmap[nb]):
                island.add(nb)
                queue.append(nb)
    return island

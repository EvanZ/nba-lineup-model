"""Small-matrix ordering shared by saved rotation serving and legacy views."""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _exposure_oriented_hclust_dendrogram(
    hierarchy: np.ndarray,
    exposure: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, list[float]]]]:
    """Render an unchanged linkage tree with its most-used branches first."""

    player_count = len(exposure)
    root = 2 * player_count - 2

    def leaves_and_exposure(node: int) -> tuple[list[int], float]:
        if node < player_count:
            return [node], float(exposure[node])
        left_node, right_node = hierarchy[node - player_count, :2].astype(int)
        left_leaves, left_exposure = leaves_and_exposure(int(left_node))
        right_leaves, right_exposure = leaves_and_exposure(int(right_node))
        if right_exposure > left_exposure:
            return right_leaves + left_leaves, right_exposure + left_exposure
        return left_leaves + right_leaves, left_exposure + right_exposure

    ordered_leaves, _ = leaves_and_exposure(root)
    positions = {leaf: 5.0 + 10.0 * index for index, leaf in enumerate(ordered_leaves)}
    maximum_height = float(hierarchy[-1, 2]) if len(hierarchy) else 0.0
    leaf_span = 10.0 * max(player_count - 1, 1)
    branches: list[dict[str, list[float]]] = []

    def branch_geometry(node: int) -> tuple[float, float]:
        if node < player_count:
            return positions[node], 0.0
        left_node, right_node = hierarchy[node - player_count, :2].astype(int)
        left_x, left_height = branch_geometry(int(left_node))
        right_x, right_height = branch_geometry(int(right_node))
        height = float(hierarchy[node - player_count, 2])
        branches.append(
            {
                "x": [
                    (left_x - 5.0) / leaf_span,
                    (left_x - 5.0) / leaf_span,
                    (right_x - 5.0) / leaf_span,
                    (right_x - 5.0) / leaf_span,
                ],
                "height": [
                    left_height / maximum_height if maximum_height else 0.0,
                    height / maximum_height if maximum_height else 0.0,
                    height / maximum_height if maximum_height else 0.0,
                    right_height / maximum_height if maximum_height else 0.0,
                ],
            }
        )
        return (left_x + right_x) / 2.0, height

    branch_geometry(root)
    return np.asarray(ordered_leaves, dtype=int), branches


def rotation_order(matrix, exposure, ordering):
    count = len(exposure)
    if count <= 2:
        return np.arange(count), []
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1)
    if ordering == "hclust":
        distance = 1 - matrix
        np.fill_diagonal(distance, 0)
        hierarchy = linkage(squareform(distance, checks=False), method="complete")
        return _exposure_oriented_hclust_dendrogram(hierarchy, exposure)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    principal = np.argsort(eigenvalues)[::-1]
    first = eigenvectors[:, principal[0]]
    if ordering == "fpc":
        return np.argsort(first), []
    second = eigenvectors[:, principal[1]]
    ratio = np.divide(second, first, out=np.zeros_like(first), where=first != 0)
    angles = np.where(first > 0, np.arctan(ratio), np.arctan(ratio) + np.pi)
    return np.argsort(angles)[::-1], []

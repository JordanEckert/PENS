from __future__ import annotations

import numpy as np

from ._metrics import distances_to
from .dendrogram import single_linkage


def centroid_nearest(pts: np.ndarray, metric: str, metric_params: dict) -> int:
    """Find the point closest to the coordinate mean of pts, the same metric being evaluated at the mean.

    Args:
        pts: (m, d) array of points, m >= 1.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Local index into pts of the nearest point; on an exact tie the smallest index.
    """
    d = distances_to(pts, pts.mean(axis=0, keepdims=True), metric, metric_params)
    return int(np.argmin(d.ravel()))


def merge_scale(X: np.ndarray, idx: np.ndarray, metric: str, metric_params: dict) -> float:
    """Compute the median single-linkage merge height of X[idx], the median over all len(idx) - 1 degree-0 death
    times, zero heights from duplicate rows included.

    Args:
        X: (n, d) array of all points.
        idx: Row indices of the points.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        The median merge height, or 0.0 when there are fewer than two points.
    """
    if len(idx) < 2:
        return 0.0
    Z = single_linkage(X[idx], metric, metric_params)
    return float(np.median(Z[:, 2]))


def farthest_point_cover(X: np.ndarray, members: np.ndarray, eps: float, metric: str, metric_params: dict) -> list[int]:
    """Cover a set of rows by farthest-point sampling (Gonzalez 1985) seeded at the row nearest its mean.

    Points are added, each the member farthest from those chosen so far, until every member lies within eps of a
    chosen point; a member at distance exactly eps counts as covered. Ties in the seed and in the farthest point go
    to the smallest position in `members`, which is the smallest row index when `members` is sorted. The result is
    an eps-net of the members: it covers them within eps and its points are pairwise more than eps apart.

    Args:
        X: (n, d) array of all points.
        members: Row indices of the rows to cover, len(members) >= 1.
        eps: Target covering radius, >= 0.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Original row indices of the chosen points, in selection order.
    """
    if len(members) == 1:
        return [int(members[0])]
    pts = X[members]
    chosen = [centroid_nearest(pts, metric, metric_params)]
    mind = distances_to(pts, pts[chosen], metric, metric_params).ravel()
    while mind.max() > eps:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        np.minimum(mind, distances_to(pts, pts[[nxt]], metric, metric_params).ravel(), out=mind)
    return [int(members[i]) for i in chosen]

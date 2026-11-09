from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist

_SCIPY_METRIC = {"manhattan": "cityblock", "l1": "cityblock", "l2": "euclidean"}


def to_scipy_metric(metric: str) -> str:
    """Map the aliases manhattan/l1 (to cityblock) and l2 (to euclidean) to their SciPy spelling.

    Args:
        metric: SciPy metric name, or one of the aliases manhattan, l1, l2.

    Returns:
        The SciPy metric name; any name that is not an alias is returned unchanged.
    """
    return _SCIPY_METRIC.get(metric, metric)


def pairwise_condensed(X: np.ndarray, metric: str, metric_params: dict) -> np.ndarray:
    """Compute condensed pairwise distances between the rows of X.

    Args:
        X: (n, d) array of points.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Condensed distance vector of length n*(n-1)/2.
    """
    return pdist(X, metric=metric, **metric_params)


def distances_to(X: np.ndarray, Y: np.ndarray, metric: str, metric_params: dict) -> np.ndarray:
    """Compute distances from every row of X to every row of Y.

    Args:
        X: (n, d) array of points.
        Y: (m, d) array of points.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        (n, m) distance matrix.
    """
    return cdist(X, Y, metric=metric, **metric_params)

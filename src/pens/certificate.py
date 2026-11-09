from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform

from ._metrics import distances_to, pairwise_condensed, to_scipy_metric


def bottleneck_bound(radii: dict) -> dict:
    """Compute the bottleneck-distance bound of Corollary 1, 2 * eps_c, for each class radius.

    Args:
        radii: Dict mapping class label to eps_c (the fitted `radii_`).

    Returns:
        Dict mapping class label to 2 * eps_c. A class the edit emptied (a guard class) has eps_c = 0 and
        receives the value 0.0 here, although no bound is claimed for it.
    """
    return {label: 2.0 * eps for label, eps in radii.items()}


def hausdorff_distance(A, B, metric="euclidean", metric_params=None) -> float:
    """Compute the Hausdorff distance between two nonempty point sets.

    Args:
        A: (m, d) array of points, m >= 1.
        B: (k, d) array of points, k >= 1.
        metric: SciPy metric name (the aliases manhattan/l1/l2 are accepted).
        metric_params: Extra keyword arguments for the metric, or None.

    Returns:
        The Hausdorff distance max(max_a d(a, B), max_b d(b, A)).
    """
    D = distances_to(np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64),
                     to_scipy_metric(metric), metric_params or {})
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def edit_cost(X, y, kept, metric="euclidean", metric_params=None) -> dict:
    """Compute the edit cost eta_c = d_H(K_c, X_c) of each class, the Hausdorff distance from the kept set to the
    observed class.

    Since K_c is a subset of X_c, this is the largest distance from a removed point of class c to K_c, and it is
    0 when the edit removed nothing from the class.

    Args:
        X: (n, d) feature array used in fit.
        y: (n,) class labels used in fit.
        kept: Dict mapping class label to kept row indices (the fitted `kept_`).
        metric: SciPy metric name (the aliases manhattan/l1/l2 are accepted).
        metric_params: Extra keyword arguments for the metric, or None.

    Returns:
        Dict mapping class label to eta_c; classes with an empty kept set are omitted, since eta_c is undefined
        for them.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    metric, metric_params = to_scipy_metric(metric), metric_params or {}
    out = {}
    for label, kept_c in kept.items():
        kept_c = np.asarray(kept_c, dtype=int)
        if len(kept_c) == 0:
            continue
        removed = np.setdiff1d(np.flatnonzero(y == label), kept_c)
        out[label] = (float(distances_to(X[removed], X[kept_c], metric, metric_params).min(axis=1).max())
                      if len(removed) else 0.0)
    return out


def end_to_end_bound(X, y, kept, radii, metric="euclidean", metric_params=None) -> dict:
    """Compute the end-to-end bound of Theorem 3 on the bottleneck distance between the diagrams of the prototypes
    and of the observed class.

    d_H(P_c, X_c) <= d_H(P_c, K_c) + d_H(K_c, X_c) <= eps_c + eta_c, so every degree satisfies
    d_B(Dgm(P_c), Dgm(X_c)) <= 2 * (eps_c + eta_c).

    Args:
        X: (n, d) feature array used in fit.
        y: (n,) class labels used in fit.
        kept: Dict mapping class label to kept row indices (the fitted `kept_`).
        radii: Dict mapping class label to eps_c (the fitted `radii_`).
        metric: SciPy metric name (the aliases manhattan/l1/l2 are accepted).
        metric_params: Extra keyword arguments for the metric, or None.

    Returns:
        Dict mapping class label to 2 * (eps_c + eta_c); classes with an empty kept set are omitted.
    """
    cost = edit_cost(X, y, kept, metric, metric_params)
    return {label: 2.0 * (float(radii[label]) + c) for label, c in cost.items()}


def verify_certificate(X, kept, prototypes, radii, maxdim=1, metric="euclidean", metric_params=None) -> dict:
    """Check the two inequalities of Corollary 1, d_H(P_c, K_c) <= eps_c and d_B(Dgm_k(P_c), Dgm_k(K_c)) <= 2 eps_c,
    numerically for every class with at least two kept points.

    Diagrams are Vietoris-Rips persistence diagrams computed by Ripser from the full distance matrix (an edge enters
    at its length), with the infinite degree-0 bar removed; bottleneck distances come from GUDHI. Both inequalities
    are tested with an absolute tolerance of 1e-9.

    Args:
        X: (n, d) feature array used in fit.
        kept: Dict mapping class label to kept row indices (the fitted `kept_`).
        prototypes: Dict mapping class label to prototype row indices (the fitted `prototypes_`).
        radii: Dict mapping class label to eps_c (the fitted `radii_`).
        maxdim: Highest homology degree to compare; degrees 0..maxdim are checked.
        metric: SciPy metric name (the aliases manhattan/l1/l2 are accepted).
        metric_params: Extra keyword arguments for the metric, or None.

    Returns:
        Dict mapping class label to a dict with keys "eps" (eps_c), "bound" (2 eps_c), "hausdorff" (d_H(P_c, K_c)),
        "bottleneck" (dict degree -> d_B) and "holds" (True when every inequality holds). Classes with fewer than two
        kept points, guard classes included, are omitted.

    Raises:
        ImportError: If ripser or gudhi is not installed.
    """
    try:
        import gudhi
        import ripser
    except ImportError as e:
        raise ImportError('verify_certificate needs ripser and gudhi: pip install "pens-tda[certificate]"') from e

    X = np.asarray(X, dtype=np.float64)
    metric, metric_params = to_scipy_metric(metric), metric_params or {}

    def diagrams(idx):
        """Compute the finite bars of the Rips diagrams of X[idx] in degrees 0..maxdim.

        Args:
            idx: Row indices of the points.

        Returns:
            List of (m_k, 2) arrays of (birth, death) pairs, one per degree, infinite bars removed.
        """
        D = squareform(pairwise_condensed(X[idx], metric, metric_params))
        dgms = ripser.ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
        return [d[np.isfinite(d[:, 1])] if len(d) else np.empty((0, 2)) for d in dgms]

    out = {}
    for label, kept_c in kept.items():
        if len(kept_c) < 2:
            continue
        protos = np.asarray(prototypes[label])
        eps = float(radii[label])
        hd = hausdorff_distance(X[kept_c], X[protos], metric, metric_params)
        dk, dp = diagrams(kept_c), diagrams(protos)
        bn = {k: float(gudhi.bottleneck_distance(dk[k], dp[k])) for k in range(maxdim + 1)}
        holds = hd <= eps + 1e-9 and all(b <= 2 * eps + 1e-9 for b in bn.values())
        out[label] = {"eps": eps, "bound": 2 * eps, "hausdorff": hd, "bottleneck": bn, "holds": bool(holds)}
    return out

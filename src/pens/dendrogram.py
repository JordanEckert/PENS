from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.cluster.hierarchy import linkage

from ._metrics import pairwise_condensed


def single_linkage(X: np.ndarray, metric: str, metric_params: dict) -> np.ndarray:
    """Build the single-linkage dendrogram of X.

    Args:
        X: (n, d) array of points, n >= 2.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Scipy linkage matrix of shape (n-1, 4).
    """
    return linkage(pairwise_condensed(X, metric, metric_params), method="single")


def node_class_counts(Z: np.ndarray, y_enc: np.ndarray, n_classes: int) -> np.ndarray:
    """Count class members under every dendrogram node.

    Args:
        Z: Linkage matrix of shape (n-1, 4).
        y_enc: (n,) integer-encoded labels.
        n_classes: Number of classes.

    Returns:
        (2n-1, n_classes) int64 array of per-node class counts.
    """
    n = len(y_enc)
    left = Z[:, 0].astype(np.int64)
    right = Z[:, 1].astype(np.int64)
    counts = np.zeros((2 * n - 1, n_classes), dtype=np.int64)
    counts[np.arange(n), y_enc] = 1
    for j in range(n - 1):
        counts[n + j] = counts[left[j]] + counts[right[j]]
    return counts


def node_purity_at_least(counts: np.ndarray, purity: float, class_totals: np.ndarray | None = None) -> np.ndarray:
    """Test exactly whether the majority-class fraction of every node, optionally after reweighting each class to
    equal total mass, is at least `purity`.

    The weights 1 / class_totals are scaled to integers by their least common multiple, and `purity` is read as the
    decimal it is written as (0.9 is 9/10), so the test runs in integers. In floating point a node of purity exactly
    0.9 can fail it: 1 / (1 + 1/9) evaluates to 0.8999999999999999.

    Args:
        counts: (2n-1, n_classes) per-node class counts.
        purity: Majority fraction to test against, in (0, 1].
        class_totals: (n_classes,) global class counts. If given, class c is weighted by 1 / class_totals[c].

    Returns:
        (2n-1,) boolean array.
    """
    p = Fraction(str(float(purity)))
    if class_totals is None:
        w = [1] * counts.shape[1]
    else:
        lcm = math.lcm(*(int(t) for t in class_totals))
        w = [lcm // int(t) for t in class_totals]
    # Python integers, so that the products cannot overflow.
    a = counts.astype(object) * np.array(w, dtype=object)
    return np.asarray(a.max(axis=1) * p.denominator >= a.sum(axis=1) * p.numerator, dtype=bool)


def accepted_leaf_assignment(
    Z: np.ndarray, counts: np.ndarray, purity: float, s_min: int, class_totals: np.ndarray | None = None
) -> np.ndarray:
    """Assign every leaf to its highest accepted ancestor node.

    Args:
        Z: Linkage matrix of shape (n-1, 4).
        counts: (2n-1, n_classes) per-node class counts.
        purity: Majority fraction at which a node is accepted.
        s_min: Nodes of this size or smaller are always accepted.
        class_totals: (n_classes,) global class counts. If given, purity is measured after weighting each
            class by the inverse of its global size.

    Returns:
        (n,) int64 array of accepted node ids, one per leaf.
    """
    n_nodes = len(counts)
    n = (n_nodes + 1) // 2
    left = Z[:, 0].astype(np.int64)
    right = Z[:, 1].astype(np.int64)
    sizes = counts.sum(axis=1)
    accept = node_purity_at_least(counts, purity, class_totals) | (sizes <= s_min)
    assign = np.full(n_nodes, -1, dtype=np.int64)
    # Decreasing ids visit every parent before its children.
    for node in range(n_nodes - 1, -1, -1):
        if assign[node] == -1 and accept[node]:
            assign[node] = node
        if node >= n and assign[node] != -1:
            assign[left[node - n]] = assign[node]
            assign[right[node - n]] = assign[node]
    return assign[:n]


def edit_clusters(
    leaf_assign: np.ndarray, y_enc: np.ndarray, class_totals: np.ndarray, min_keep: int, balanced: bool = False
) -> list[tuple[int, np.ndarray]]:
    """Give each accepted cluster to its majority class and drop the minority members.

    Args:
        leaf_assign: (n,) accepted node id per leaf.
        y_enc: (n,) integer-encoded labels.
        class_totals: (n_classes,) global class counts.
        min_keep: Clusters keeping fewer points than this are dropped.
        balanced: If True, the majority is taken after weighting each class by the inverse of its global size.

    Returns:
        List of (class_index, kept_indices) for the surviving clusters.
    """
    n_classes = len(class_totals)
    clusters: list[tuple[int, np.ndarray]] = []
    for cid in np.unique(leaf_assign):
        members = np.flatnonzero(leaf_assign == cid)
        cvec = np.bincount(y_enc[members], minlength=n_classes)
        if balanced:
            cvec = cvec / class_totals
        cands = np.flatnonzero(cvec == cvec.max())
        # Ties go to the globally rarer class, then the smaller class index.
        maj = int(cands[np.lexsort((cands, class_totals[cands]))[0]])
        kept = members[y_enc[members] == maj]
        if len(kept) >= min_keep:
            clusters.append((maj, kept))
    return clusters


def descend(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    class_totals: np.ndarray,
    purity: float,
    s_min: int,
    min_keep: int,
    metric: str,
    metric_params: dict,
    balanced: bool = False,
) -> list[tuple[int, np.ndarray]]:
    """Cut the joint single-linkage dendrogram by purity and edit each cluster.

    Args:
        X: (n, d) array of points, n >= 2.
        y_enc: (n,) integer-encoded labels.
        n_classes: Number of classes.
        class_totals: (n_classes,) global class counts.
        purity: Majority fraction at which a node is accepted.
        s_min: Nodes of this size or smaller are always accepted.
        min_keep: Clusters keeping fewer points than this are dropped.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.
        balanced: If True, purity and the majority claim weight each class by the inverse of its global size.

    Returns:
        List of (class_index, kept_indices) for the surviving clusters.
    """
    Z = single_linkage(X, metric, metric_params)
    counts = node_class_counts(Z, y_enc, n_classes)
    leaf_assign = accepted_leaf_assignment(Z, counts, purity, s_min, class_totals if balanced else None)
    return edit_clusters(leaf_assign, y_enc, class_totals, min_keep, balanced)

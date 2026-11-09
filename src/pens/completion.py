from __future__ import annotations

import numpy as np

from ._metrics import distances_to
from .net import centroid_nearest

_BLOCK = 512


def balanced_weights(y_T: np.ndarray, n_classes: int) -> np.ndarray:
    """Weight every point by the inverse size of its class, scaled so that the mean weight is 1.

    Args:
        y_T: (m,) integer-encoded labels of the target points.
        n_classes: Number of classes.

    Returns:
        (m,) array of weights; every class present receives total weight m / (number of classes present).
    """
    cnt = np.bincount(y_T, minlength=n_classes).astype(float)
    return len(y_T) / ((cnt > 0).sum() * cnt[y_T])


class _Nearest:
    """Nearest prototype of every target point; on a distance tie the prototype added first is kept.

    Args:
        X: (n, d) array of all points.
        T: Row indices of the target points.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Attributes:
        dist: (len(T),) distance from every target point to its nearest prototype (inf before any is added).
        proto: (len(T),) row index of that prototype (-1 before any is added).
    """

    def __init__(self, X: np.ndarray, T: np.ndarray, metric: str, metric_params: dict):
        self.X, self.XT = X, X[T]
        self.metric, self.metric_params = metric, metric_params
        self.dist = np.full(len(T), np.inf)
        self.proto = np.full(len(T), -1, dtype=np.int64)

    def add(self, protos) -> None:
        """Add prototypes and update the nearest prototype of every target point.

        Args:
            protos: Row indices of the prototypes to add, in insertion order; a prototype replaces the current
                nearest one only when it is strictly closer.

        Returns:
            None.
        """
        protos = np.asarray(protos, dtype=np.int64)
        for a in range(0, len(protos), _BLOCK):
            block = protos[a:a + _BLOCK]
            D = distances_to(self.XT, self.X[block], self.metric, self.metric_params)
            for j, p in enumerate(block):
                better = D[:, j] < self.dist
                self.dist[better] = D[better, j]
                self.proto[better] = p


def consistent_completion(
    X: np.ndarray, y_enc: np.ndarray, T: np.ndarray, S0, metric: str, metric_params: dict
) -> list[int]:
    """Add target points to S0 until every target point has a nearest prototype of its own class.

    This is the FCNN1 rule of Angiulli (2007) started from S0: in every round, each prototype receives the
    misclassified target point of its Voronoi cell that is nearest to it (ties to the smaller row index). A target
    point counts as misclassified when its nearest prototype, ties going to the prototype added first, carries
    another label; on an exact distance tie the 1-NN prediction can still depend on the classifier's tie rule.

    Args:
        X: (n, d) array of all points.
        y_enc: (n,) integer-encoded labels.
        T: Row indices of the target points (the kept points).
        S0: Row indices of the initial prototypes.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Row indices of the prototypes, S0 first and then the added points in order.
    """
    T = np.asarray(T)
    S = [int(p) for p in S0]
    in_S = np.zeros(len(X), dtype=bool)
    in_S[S] = True
    yT = y_enc[T]
    near = _Nearest(X, T, metric, metric_params)
    near.add(S)
    while True:
        mis = (y_enc[near.proto] != yT) & ~in_S[T]
        if not mis.any():
            return S
        idx = np.flatnonzero(mis)
        protos, dists = near.proto[idx], near.dist[idx]
        order = np.lexsort((idx, dists, protos))
        first = np.ones(len(order), dtype=bool)
        first[1:] = protos[order][1:] != protos[order][:-1]
        new = T[idx[order][first]]
        S.extend(int(p) for p in new)
        in_S[new] = True
        near.add(new)


def gain_completion(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    T: np.ndarray,
    S0,
    gain: float,
    weights: np.ndarray,
    metric: str,
    metric_params: dict,
) -> list[int]:
    """Add target points to S0 only where they correct the classification of other target points.

    In every round the misclassified target points are grouped by (nearest prototype, label), and each group
    proposes its member nearest to the group mean. A proposal is accepted if adding it raises the weighted
    number of correctly classified target points, not counting itself, by at least `gain`. Proposals accepted
    in the same round have disjoint sets of target points that would switch to them. The rounds stop when no
    proposal qualifies, so an isolated misclassified point, which can only correct itself, is never added.

    Args:
        X: (n, d) array of all points.
        y_enc: (n,) integer-encoded labels.
        n_classes: Number of classes.
        T: Row indices of the target points (the kept points).
        S0: Row indices of the initial prototypes.
        gain: Minimum weighted net number of other target points a proposal must correct.
        weights: (len(T),) weight of every target point.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Row indices of the prototypes, S0 first and then the added points in order.
    """
    T = np.asarray(T)
    S = [int(p) for p in S0]
    in_S = np.zeros(len(X), dtype=bool)
    in_S[S] = True
    yT = y_enc[T]
    near = _Nearest(X, T, metric, metric_params)
    near.add(S)
    while True:
        correct = y_enc[near.proto] == yT
        mis = ~correct & ~in_S[T]
        if not mis.any():
            return S
        idx = np.flatnonzero(mis)
        key = near.proto[idx] * n_classes + yT[idx]
        order = np.lexsort((idx, key))
        idx, key = idx[order], key[order]
        groups = np.split(idx, np.flatnonzero(np.diff(key)) + 1)
        cands = np.array([grp[centroid_nearest(X[T[grp]], metric, metric_params)] for grp in groups])
        gains = np.empty(len(cands))
        reach = {}
        for a in range(0, len(cands), _BLOCK):
            cb = cands[a:a + _BLOCK]
            infl = distances_to(X[T[cb]], near.XT, metric, metric_params) < near.dist[None, :]
            same = yT[cb][:, None] == yT[None, :]
            delta = (same & ~correct[None, :]).astype(float) - (~same & correct[None, :]).astype(float)
            gb = (infl * delta * weights[None, :]).sum(axis=1) - weights[cb]
            gains[a:a + _BLOCK] = gb
            for i in np.flatnonzero(gb >= gain - 1e-12):
                reach[a + int(i)] = infl[i]
        ok = np.flatnonzero(gains >= gain - 1e-12)
        if len(ok) == 0:
            return S
        touched = np.zeros(len(T), dtype=bool)
        new = []
        for i in ok[np.lexsort((cands[ok], -gains[ok]))]:
            if touched[reach[int(i)]].any():
                continue
            touched |= reach[int(i)]
            new.append(int(T[cands[i]]))
        S.extend(new)
        in_S[new] = True
        near.add(new)


def prune_completion(
    X: np.ndarray,
    y_enc: np.ndarray,
    T: np.ndarray,
    S,
    protected: set,
    weights: np.ndarray,
    metric: str,
    metric_params: dict,
) -> list[int]:
    """Remove unprotected prototypes whose removal does not lower the weighted number of correct target points.

    Candidates are visited from the interior outwards, by decreasing distance to the nearest target point of
    another class (as in DROP3), with the smallest index first on ties. Only the target points currently served
    by a candidate can change their nearest prototype, so only those are re-examined.

    Args:
        X: (n, d) array of all points.
        y_enc: (n,) integer-encoded labels.
        T: Row indices of the target points (the kept points).
        S: Row indices of the prototypes.
        protected: Row indices that are never removed (the certified core).
        weights: (len(T),) weight of every target point.
        metric: Scipy metric name.
        metric_params: Extra keyword arguments for the metric.

    Returns:
        Sorted row indices of the remaining prototypes.
    """
    T = np.asarray(T)
    S = np.array(sorted(S), dtype=np.int64)
    yT = y_enc[T]
    nn = np.empty(len(T), dtype=np.int64)
    for a in range(0, len(T), _BLOCK):
        nn[a:a + _BLOCK] = S[np.argmin(distances_to(X[T[a:a + _BLOCK]], X[S], metric, metric_params), axis=1)]
    correct = y_enc[nn] == yT
    alive = {int(p) for p in S}
    cands = np.array([int(p) for p in S if int(p) not in protected], dtype=np.int64)
    if len(cands) == 0:
        return sorted(alive)
    enemy = np.full(len(cands), np.inf)
    for c in np.unique(y_enc[cands]):
        sel = np.flatnonzero(y_enc[cands] == c)
        other = T[yT != c]
        if len(other):
            for a in range(0, len(sel), _BLOCK):
                rows = sel[a:a + _BLOCK]
                enemy[rows] = distances_to(X[cands[rows]], X[other], metric, metric_params).min(axis=1)
    for p in cands[np.lexsort((cands, -enemy))]:
        p = int(p)
        aff = np.flatnonzero(nn == p)
        rest = np.array(sorted(alive - {p}), dtype=np.int64)
        if len(rest) == 0:
            continue
        if len(aff):
            new = rest[np.argmin(distances_to(X[T[aff]], X[rest], metric, metric_params), axis=1)]
            new_correct = y_enc[new] == yT[aff]
            if (weights[aff] * new_correct).sum() < (weights[aff] * correct[aff]).sum() - 1e-12:
                continue
            nn[aff] = new
            correct[aff] = new_correct
        alive.discard(p)
    return sorted(alive)

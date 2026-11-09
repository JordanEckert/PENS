import random

import numpy as np
from scipy.spatial.distance import cdist, squareform

from vendor.spotgreedy import SPOT_GreedySubsetSelection
from vendor.waashk_ib3 import IB3

KEEL_SEED = 12345678


def _encode(y):
    """Encode labels as class indices 0..C-1 in sorted label order.

    Args:
        y: (n,) class labels.

    Returns:
        Tuple (y_enc, C): the (n,) integer class codes and the number of classes C.
    """
    classes, y_enc = np.unique(y, return_inverse=True)
    return y_enc, len(classes)


def _keel_knn(D, k):
    """KEEL's k nearest neighbors of every row of D.

    Entries equal to 0 (the query itself or an exact duplicate) are skipped and distance ties go to the earlier
    column, which is KEEL's nearest-neighbor rule.

    Args:
        D: (n_rows, n_cols) distance matrix; rows are queries, columns candidates in order.
        k: Number of neighbors.

    Returns:
        (n_rows, k) integer array of column indices, -1 where fewer than k candidates exist.
    """
    D = np.where(D > 0, D, np.inf)
    rows = np.arange(len(D))
    nn = np.full((len(D), k), -1, dtype=np.int64)
    for j in range(min(k, D.shape[1])):
        col = D.argmin(axis=1)
        ok = np.isfinite(D[rows, col])
        nn[ok, j] = col[ok]
        D[rows[ok], col[ok]] = np.inf
    return nn


def _keel_vote(nn, labels, n_classes):
    """Majority class over the neighbors in nn; ties and empty votes go to the smaller class index (KEEL's rule).

    Args:
        nn: (n_rows, k) neighbor column indices from _keel_knn, -1 for none.
        labels: (n_cols,) class codes of the candidates.
        n_classes: Number of classes C.

    Returns:
        (n_rows,) integer array of predicted class codes.
    """
    votes = np.zeros((len(nn), n_classes), dtype=np.int64)
    for j in range(nn.shape[1]):
        ok = np.flatnonzero(nn[:, j] >= 0)
        np.add.at(votes, (ok, labels[nn[ok, j]]), 1)
    return votes.argmax(axis=1)


def _keel_randint(rs, low, high):
    """KEEL's Randomize.Randint: an integer in [low, high), or low when low == high.

    Args:
        rs: numpy RandomState (MT19937 seeded with init_genrand, which is KEEL's generator).
        low: Lower bound, inclusive.
        high: Upper bound, exclusive.

    Returns:
        The drawn integer.
    """
    return int(low + (high - low) * rs.random_sample())


def _enn_mask(D, y, n_classes, k):
    """Mask of the points whose k nearest neighbors (KEEL rule) vote for their own class.

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        n_classes: Number of classes.
        k: Number of neighbors.

    Returns:
        (n,) boolean array, True for the points ENN keeps.
    """
    return _keel_vote(_keel_knn(D, k), y, n_classes) == y


def enn(X, y, k=3):
    """Wilson editing (KEEL ENN): keep the points whose k nearest neighbors vote for their own class.

    Port of KEEL (github.com/SCI2SUGR/KEEL, commit 5f404a9, src/keel/Algorithms/Preprocess/Instance_Selection) with
    KEEL's nearest-neighbor rule: a point at distance 0 from the query (the query itself or an exact duplicate) is
    never its neighbor, distance ties go to the earlier point and vote ties to the smaller class index.
    verify_baselines.py checks that it selects the same rows as KEEL.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        k: Number of neighbors.

    Returns:
        Sorted integer array of row indices into X of the kept points.
    """
    y, n_classes = _encode(y)
    return np.flatnonzero(_enn_mask(cdist(X, X), y, n_classes, k))


def cnn(X, y, k=1, seed=KEEL_SEED):
    """Hart's condensed nearest neighbor (KEEL CNN).

    Port of KEEL (see enn) with KEEL's nearest-neighbor rule, drawing from KEEL's random generator (MT19937 seeded
    with init_genrand, which is numpy's RandomState). The store starts with one point per class: a random position,
    moved forward cyclically to the first point of the class. Each pass visits all points in a fresh random order and
    adds every point outside the store that k-NN on the store misclassifies; passes repeat until one adds nothing.
    As in KEEL, the store is kept sorted by index except that the point added at the previous step sits at the end
    until the next step, which matters only for exact distance ties. verify_baselines.py checks it against KEEL.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        k: Number of neighbors.
        seed: Seed of the random generator (KEEL's default KEEL_SEED).

    Returns:
        Sorted integer array of row indices into X of the store.
    """
    y, n_classes = _encode(y)
    n = len(X)
    rs = np.random.RandomState(seed)
    S = []
    for c in range(n_classes):
        pos, cont = _keel_randint(rs, 0, n - 1), 0
        while y[pos] != c and cont < n:
            pos, cont = (pos + 1) % n, cont + 1
        if cont < n:
            S.append(pos)
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    D = cdist(X, X)
    is_sorted = S == sorted(S)
    conj, dirty = None, True
    changed = True
    while changed:
        changed = False
        order = list(range(n))
        for i in range(n):
            pos = _keel_randint(rs, i, n - 1)
            order[i], order[pos] = order[pos], order[i]
        for q in order:
            if dirty:
                conj, dirty = np.array(S, dtype=np.int64), False
            if not is_sorted:
                S.sort()
                is_sorted, dirty = True, True
            if in_S[q]:
                continue
            pred = _keel_vote(_keel_knn(D[q:q + 1, conj], k), y[conj], n_classes)[0]
            if pred != y[q]:
                S.append(q)
                in_S[q] = True
                is_sorted, dirty, changed = False, True, True
    return np.flatnonzero(in_S)


def _vote(labels):
    """Majority label of a distance-ordered label list; count ties go to the tied class seen first (nearest).

    Args:
        labels: Nonempty list of labels, ordered by increasing distance to the query.

    Returns:
        The winning label.
    """
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    top = max(counts.values())
    for lab in labels:
        if counts[lab] == top:
            return lab


def drop3(X, y, k=3):
    """DROP3 (Wilson & Martinez 2000).

    1. ENN (k) noise filter; the filtered set T' is the reference set below.
    2. Visit instances of T' by distance to nearest enemy in T', farthest first.
    3. Remove P from S if at least as many of its associates are classified correctly without P as with P. Every
       instance of T' (removed or not, as in DROP2) keeps a list of its k+1 nearest neighbors in the current S;
       "with P" votes over the first k of that list, "without P" over the list minus P. When P is removed each
       associate drops P, takes its next nearest instance still in S, and becomes an associate of it.
    verify_baselines.py checks it against a naive version that recomputes the neighbor lists at every step, and runs
    the waashk/instanceselection implementation next to it.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        k: Number of neighbors.

    Returns:
        Sorted integer array of row indices into X of the retained instances.
    """
    y_enc, n_classes = _encode(y)
    T = np.flatnonzero(_enn_mask(cdist(X, X), y_enc, n_classes, k))
    yt = y_enc[T]
    m = len(T)
    D = cdist(X[T], X[T])

    enemy = np.full(m, np.inf)
    for c in np.unique(yt):
        same = yt == c
        if (~same).any():
            enemy[same] = D[np.ix_(same, ~same)].min(axis=1)
    visit = np.argsort(-enemy, kind="stable")

    # Per-row neighbor order; self is pushed to the last position.
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable").astype(np.int32)
    del D

    n_others = m - 1
    width = min(k + 1, n_others)
    alive = np.ones(m, dtype=bool)
    ptr = [width] * m  # next unread position in order[a]
    nbrs = [order[a, :width].tolist() for a in range(m)]
    assoc = [set() for _ in range(m)]
    for a in range(m):
        for b in nbrs[a]:
            assoc[b].add(a)
    yl = yt.tolist()

    def correct(a, exclude):
        """Whether k-NN over a's current neighbor list, less `exclude`, classifies a correctly.

        Args:
            a: Local index into T of the instance.
            exclude: Local index to leave out of the vote ("without P"), or None ("with P").

        Returns:
            True if the vote over the first k remaining neighbors equals a's label; False for an empty list.
        """
        labs = [yl[b] for b in nbrs[a] if b != exclude][:k]
        return bool(labs) and _vote(labs) == yl[a]

    for p in visit.tolist():
        A = assoc[p]
        n_with = sum(correct(a, None) for a in A)
        n_without = sum(correct(a, p) for a in A)
        if n_without >= n_with:
            alive[p] = False
            for a in A:
                nbrs[a].remove(p)
                q = ptr[a]
                row = order[a]
                while q < n_others and not alive[row[q]]:
                    q += 1
                if q < n_others:
                    new = int(row[q])
                    nbrs[a].append(new)
                    assoc[new].add(a)
                    q += 1
                ptr[a] = q
            assoc[p] = set()
    return T[alive]


def icf(X, y, k=3):
    """Iterative case filtering (Brighton & Mellish 2002; KEEL ICF).

    Port of KEEL (see enn) with KEEL's nearest-neighbor rule in the ENN step. Nearest-enemy distances are computed
    once on the whole training set. After ENN (k), the coverage of x is the number of points of S closer to x than
    x's nearest enemy, and its reachability the number of points of S that have x closer than their own nearest
    enemy. Every x with reachability above coverage is removed at once, and this repeats until nothing is removed.
    verify_baselines.py checks it against KEEL.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        k: Number of neighbors of the ENN step.

    Returns:
        Sorted integer array of row indices into X of the retained points.
    """
    y, n_classes = _encode(y)
    D = cdist(X, X)
    enemy = np.where(y[:, None] != y[None, :], D, np.inf).min(axis=1)
    keep = _enn_mask(D, y, n_classes, k)
    adaptable = D < enemy[:, None]  # adaptable[x, i]: i is closer to x than x's nearest enemy
    np.fill_diagonal(adaptable, False)
    del D
    while True:
        coverage = adaptable[:, keep].sum(axis=1)
        reachable = adaptable[keep, :].sum(axis=0)
        drop = keep & (reachable > coverage)
        if not drop.any():
            return np.flatnonzero(keep)
        keep &= ~drop


def hmnei(X, y, epsilon=0.1):
    """Hit miss network editing, iterated (Marchiori 2008; KEEL HMNEI).

    Port of KEEL (see enn) with KEEL's nearest-neighbor rule. Each round builds the hit miss network of the current
    set S: every point points to its nearest other point of each class, a hit when the class is its own and a miss
    otherwise. With w_c the share of class c in S, a point is removed if w_c * misses + epsilon > (1 - w_c) * hits
    (R1); it is restored if its class keeps fewer than 4 points and it has an incoming edge (R2), if there are more
    than 3 classes, it has an incoming edge and fewer than C // 2 misses (R3), or if it has at least
    (class size in S) // 4 hits (R4). Rounds continue while 1-NN on the new set classifies at least as many training
    points correctly and the set shrinks; the result is the set at the start of the last round. verify_baselines.py
    checks it against KEEL.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        epsilon: Slack of the removal rule R1.

    Returns:
        Sorted integer array of row indices into X of the retained points.
    """
    y, n_classes = _encode(y)
    D = cdist(X, X)
    n = len(X)
    rows_all = np.arange(n)
    S, correct = np.arange(n), 0
    while True:
        correct_before = correct
        yS = y[S]
        m = len(S)
        DS = D[np.ix_(S, S)]
        np.fill_diagonal(DS, np.inf)
        rows = np.arange(m)
        hit = np.zeros(m, dtype=np.int64)
        miss = np.zeros(m, dtype=np.int64)
        for c in range(n_classes):
            cols = np.flatnonzero(yS == c)
            if len(cols) == 0:
                continue
            sub = DS[:, cols]
            j = sub.argmin(axis=1)
            ok = np.isfinite(sub[rows, j])
            target = cols[j[ok]]
            is_hit = yS[ok] == c
            np.add.at(hit, target[is_hit], 1)
            np.add.at(miss, target[~is_hit], 1)
        del DS, sub
        size = np.bincount(yS, minlength=n_classes)
        w = size[yS] / m
        marks = ~(w * miss + epsilon > (1 - w) * hit)
        edges = hit + miss > 0
        for c in range(n_classes):
            if np.count_nonzero(marks & (yS == c)) < 4:
                marks |= (yS == c) & edges
        if n_classes > 3:
            marks |= edges & (miss < n_classes // 2)
        marks |= hit >= size[yS] // 4
        S_new = S[marks]
        if len(S_new):
            DT = D[:, S_new]
            DT = np.where(DT > 0, DT, np.inf)
            j = DT.argmin(axis=1)
            pred = np.where(np.isfinite(DT[rows_all, j]), y[S_new][j], 0)
            del DT
        else:
            pred = np.zeros(n, dtype=np.int64)
        correct = int(np.count_nonzero(pred == y))
        if not (correct >= correct_before and len(S_new) < m):
            return S
        S = S_new


def ib3(X, y, accept=0.9, drop=0.7, seed=0):
    """IB3 (Aha et al. 1991), running the vendored waashk/instanceselection implementation (vendor/waashk_ib3.py).

    The implementation follows KEEL's; a stored instance is acceptable when accept * accuracy > class frequency and
    is dropped when accuracy < drop * class frequency. Random choices come from random.Random(seed).
    verify_baselines.py checks it against KEEL (with KEEL's generator in place of random.Random).

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        accept: Acceptance level.
        drop: Drop level.
        seed: Seed of random.Random.

    Returns:
        Sorted integer array of row indices into X of the accepted instances.
    """
    y, _ = _encode(y)
    m = IB3(rng=random.Random(seed))
    m.nAccept, m.nDrop = accept, drop
    return np.sort(np.asarray(m.fit(X, y).sample_indices_, dtype=np.int64))


def _local_sets(D, y):
    """Local sets (Leyva et al. 2015) of every point, with its nearest enemy.

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.

    Returns:
        Tuple (ls, enemy, has_enemy): ls is an (n, n) boolean array with ls[i, j] True if j != i has the class of i
        and is closer to i than i's nearest enemy; enemy is the (n,) index of each point's nearest enemy (first
        index on ties); has_enemy is the (n,) boolean mask of the points that have one.
    """
    E = np.where(y[:, None] != y[None, :], D, np.inf)
    rows = np.arange(len(D))
    enemy = E.argmin(axis=1)
    enemy_d = E[rows, enemy]
    ls = (y[:, None] == y[None, :]) & (D < enemy_d[:, None])
    np.fill_diagonal(ls, False)
    return ls, enemy, np.isfinite(enemy_d)


def lsbo(X, y):
    """Local set border selector (Leyva et al. 2015): LSSm followed by LSBo.

    Port of dpr1005/Semisupervised-learning-and-instance-selection-methods. LSSm keeps the points whose usefulness
    (number of local sets containing them) is at least their harmfulness (number of points of which they are the
    nearest enemy). On the kept points, local sets are recomputed and points are visited by increasing local-set
    size (smaller index first on ties); a point is selected if its local set contains no selected point.
    verify_baselines.py checks it against dpr1005. The method has a single configuration.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.

    Returns:
        Sorted integer array of row indices into X of the selected points.
    """
    y, _ = _encode(y)
    D = cdist(X, X)
    ls, enemy, has_enemy = _local_sets(D, y)
    useful = ls.sum(axis=0)
    harmful = np.bincount(enemy[has_enemy], minlength=len(X))
    S1 = np.flatnonzero(useful >= harmful)
    ls1, _, _ = _local_sets(D[np.ix_(S1, S1)], y[S1])
    chosen = np.zeros(len(S1), dtype=bool)
    for e in np.argsort(ls1.sum(axis=1), kind="stable"):
        if not (ls1[e] & chosen).any():
            chosen[e] = True
    return S1[chosen]


def _ps_greedy(D, y, eps, own_class_only=True):
    """Greedy prototype selection of Bien and Tibshirani (2011, Section 3.2) on a distance matrix.

    Balls are B(x_j) = {x : d(x, x_j) < eps}; the pair with the largest
    dObj = (points of the class newly covered) - (points of other classes in the ball) - lambda, lambda = 1/n, is
    added while dObj > 0. The integer program decouples by class, so each class is run on its own, and ties go to the
    smaller candidate index.

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        eps: Ball radius.
        own_class_only: If True, a point is a candidate prototype for its own class only; if False, every point is a
            candidate for every class, as in the paper and in their R package protoclass.

    Returns:
        List of (candidate index, class code) pairs in the order they are added.
    """
    pairs = []
    ball = D < eps
    for c in np.unique(y):
        own = np.flatnonzero(y == c)
        cand = own if own_class_only else np.arange(len(y))
        eta = ball[np.ix_(cand, np.flatnonzero(y != c))].sum(axis=1)   # points of other classes in each ball
        cover = ball[np.ix_(cand, own)]                                  # points of class c in each ball
        xi = cover.sum(axis=1).astype(np.int64)                          # points of class c newly covered
        uncovered = np.ones(len(own), dtype=bool)
        while True:
            gain = xi - eta
            j = int(np.argmax(gain))
            if gain[j] < 1:                       # dObj = gain - 1/n > 0 iff gain >= 1 (integers)
                break
            pairs.append((int(cand[j]), int(c)))
            new = np.flatnonzero(cover[j] & uncovered)
            uncovered[new] = False
            xi -= cover[:, new].sum(axis=1)
    return pairs


def ps(X, y, q=0.02):
    """Prototype selection by prize-collecting set cover, greedy (Bien and Tibshirani 2011).

    eps is the q-quantile of the pairwise distances of X. The paper lets any training point be a prototype of any
    class; here a point can only be a prototype of its own class, so that the selection is a subset of the training
    rows with their labels. verify_baselines.py checks the unrestricted greedy against their R package protoclass and
    compares it with the own-class version.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        q: Quantile of the pairwise distances that sets eps.

    Returns:
        Sorted integer array of row indices into X of the prototypes.
    """
    y_enc, _ = _encode(y)
    D = cdist(X, X)
    eps = float(np.quantile(squareform(D, checks=False), q)) if len(X) > 1 else 0.0
    return np.sort(np.asarray([j for j, _ in _ps_greedy(D, y_enc, eps)], dtype=np.int64))


def _min_vertex_cover(D, y, gamma):
    """Points to remove so that no two points of different classes are closer than gamma.

    A minimum vertex cover of the conflict graph (pairs of different classes closer than gamma), exact for two
    classes (Konig's theorem on a maximum bipartite matching) and greedy by degree for more (ties to the smaller
    index).

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        gamma: Distance threshold of the conflict graph.

    Returns:
        Sorted integer array of the row indices to remove (empty when there is no conflict).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    conflict = (D < gamma) & (y[:, None] != y[None, :])
    if not conflict.any():
        return np.empty(0, dtype=np.int64)
    classes = np.unique(y)
    if len(classes) == 2:
        L, R = np.flatnonzero(y == classes[0]), np.flatnonzero(y == classes[1])
        G = csr_matrix(conflict[np.ix_(L, R)])
        match = maximum_bipartite_matching(G, perm_type="column")  # match[l] = matched column or -1
        matched_r = np.full(len(R), -1)
        matched_r[match[match >= 0]] = np.flatnonzero(match >= 0)
        # Konig: Z = vertices reachable from unmatched left vertices by alternating paths; cover = (L \ Z) + (R & Z).
        zl, zr = match < 0, np.zeros(len(R), dtype=bool)
        frontier = np.flatnonzero(zl)
        while len(frontier):
            reach = np.unique(G[frontier].indices)
            new_r = reach[~zr[reach]]
            zr[new_r] = True
            back = matched_r[new_r]
            back = back[(back >= 0) & ~zl[back]]
            zl[back] = True
            frontier = back
        cover = np.concatenate([L[~zl], R[zr]])
        assert len(cover) == int((match >= 0).sum())
        return np.sort(cover)
    degree = conflict.sum(axis=1)
    removed = []
    while degree.max() > 0:
        v = int(np.argmax(degree))
        removed.append(v)
        nb = np.flatnonzero(conflict[v])
        conflict[v, nb] = conflict[nb, v] = False
        degree[nb] -= 1
        degree[v] = 0
    return np.sort(np.asarray(removed, dtype=np.int64))


def gkn(X, y, q=0.0):
    """Margin-based sample compression (Gottlieb, Kontorovich and Nisnevitch 2014).

    1. gamma is the q-quantile of the distances from every point to its nearest point of another class (q = 0 gives
       the margin of the sample). The points of a minimum vertex cover of the conflict graph are removed (pairs of
       different classes closer than gamma; Section 3), so that the rest S has margin at least gamma.
    2. Algorithm 1: a net of S at its margin, built greedily in index order (a point joins if it is at least the
       margin away from every chosen point); it is consistent with S.
    3. Algorithm 2: the consistent pruning heuristic, with distances scaled by the diameter of S. It runs over the
       scales 2^i down to the margin, so it is skipped when the margin is 0, which happens when gamma is 0 and points
       of different classes coincide (page_blocks at q = 0); the net is then all of S.
    The structural risk minimization of Section 3 needs the doubling dimension; q is tuned instead, like the
    parameters of the other selectors. verify_baselines.py checks the result against a literal transcription of the
    two algorithms and checks the consistency that the paper proves.

    Args:
        X: (n, d) min-max scaled training features (Euclidean metric).
        y: (n,) class labels.
        q: Quantile of the nearest-enemy distances that sets gamma (0 gives the margin of the sample).

    Returns:
        Sorted integer array of row indices into X of the pruned net.
    """
    y_enc, _ = _encode(y)
    D = cdist(X, X)
    other = y_enc[:, None] != y_enc[None, :]
    nearest_enemy = np.where(other, D, np.inf).min(axis=1)
    finite = nearest_enemy[np.isfinite(nearest_enemy)]
    gamma = float(np.quantile(finite, q)) if len(finite) else 0.0
    S = np.setdiff1d(np.arange(len(X)), _min_vertex_cover(D, y_enc, gamma))
    DS, yS = D[np.ix_(S, S)], y_enc[S]
    opp = yS[:, None] != yS[None, :]
    margin = float(DS[opp].min()) if opp.any() else np.inf
    # Algorithm 1.
    net = [0]
    to_net = DS[0].copy()
    for p in range(1, len(S)):
        if to_net[p] >= margin:
            net.append(p)
            np.minimum(to_net, DS[p], out=to_net)
    # Algorithm 2.
    diam = float(DS.max()) if len(S) > 1 else 0.0
    if np.isfinite(margin) and margin > 0:
        g = margin / diam
        alive = np.zeros(len(S), dtype=bool)
        alive[net] = True
        for i in range(0, int(np.ceil(np.log2(g))) - 1, -1):  # i = 0, -1, ..., ceil(log2 g)
            r = 2.0 ** i
            for p in net:
                if not alive[p]:
                    continue
                enemies = np.flatnonzero(alive & (yS != yS[p]))
                if len(enemies) and DS[p, enemies].min() / diam < 2 * r:
                    continue
                drop = alive & (DS[p] / diam < r - g)
                drop[p] = False
                alive &= ~drop
        net = [p for p in net if alive[p]]
    return np.sort(S[net])


def random_matched(y, counts, rng):
    """Size-matched random baseline: counts[c] points of each class c, uniformly without replacement.

    Args:
        y: (n,) class labels.
        counts: Dict mapping class label c to the number of points to draw from class c (the number of PENS
            prototypes of that class).
        rng: numpy Generator used for the draws.

    Returns:
        Sorted integer array of row indices into X of the drawn points.
    """
    parts = [rng.choice(np.flatnonzero(y == c), size=kc, replace=False) for c, kc in counts.items()]
    return np.sort(np.concatenate(parts))


def spot_matched(X, y, counts):
    """Size-matched SPOTgreedy baseline (Gurumoorthy et al. 2021), run on each class.

    Uses the implementation vendored from InterpretML (vendor/spotgreedy.py): counts[c] points of class c whose
    empirical distribution is closest in optimal transport (Euclidean cost, uniform weights) to that of the whole
    class.

    Args:
        X: (n, d) min-max scaled training features.
        y: (n,) class labels.
        counts: Dict mapping class label c to the number of points to select from class c (the number of PENS
            prototypes of that class).

    Returns:
        Sorted integer array of row indices into X of the selected points.
    """
    parts = []
    for c, kc in counts.items():
        idx = np.flatnonzero(y == c)
        sel, _ = SPOT_GreedySubsetSelection(cdist(X[idx], X[idx]), np.ones(len(idx)) / len(idx), kc)
        parts.append(idx[sel])
    return np.sort(np.concatenate(parts))

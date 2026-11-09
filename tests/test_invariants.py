import numpy as np
import pytest

import pens.selector as selector
from pens import PENS, edit_cost
from pens._metrics import distances_to
from pens.dendrogram import accepted_leaf_assignment, node_class_counts, node_purity_at_least, single_linkage
from pens.net import farthest_point_cover, merge_scale

# --------------------------------------------------------------------------------------------------------------
# Shared checkers
# --------------------------------------------------------------------------------------------------------------


def _as_int(a):
    return np.asarray(a, dtype=int)


def assert_lemma2(model, X, y):
    """Assert Lemma 2 (a)-(c) for a fitted PENS model.

    (a) The kept sets of the accepted clusters are pairwise disjoint, every kept row carries the label of the
        class that claims it, and the per-class kept sets are exactly the union of the clusters.
    (b) For every class with a nonempty kept set K_c, the core Q_c is an eps_c-net of K_c: Q_c covers K_c
        within eps_c and distinct points of Q_c are strictly more than eps_c apart. eps_c is 0 when K_c is a
        singleton and the shared pooled radius otherwise.
    (c) Q_c is a subset of the prototypes P_c, which is a subset of K_c. When K_c is empty, P_c and Q_c are the
        single guard row tau_c, the row of X_c nearest the class mean (smaller row index on ties), with
        eps_c = 0. Every class has at least one prototype.

    Distances are computed with `pens._metrics.distances_to`, the function the fit uses, so covering and
    separation are compared exactly.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    metric, params = model.metric, model.metric_params
    classes = list(model.classes_)
    np.testing.assert_array_equal(model.classes_, np.unique(y))

    # (a) partition of the kept rows and class membership
    cluster_rows = [_as_int(k) for _, k in model.clusters_]
    cluster_union = np.concatenate(cluster_rows) if cluster_rows else np.empty(0, dtype=int)
    assert len(np.unique(cluster_union)) == len(cluster_union), "accepted clusters are not disjoint"
    for label, kept in model.clusters_:
        assert label in classes
        assert np.all(y[_as_int(kept)] == label)
    kept_union = np.concatenate([_as_int(model.kept_[c]) for c in classes])
    assert len(np.unique(kept_union)) == len(kept_union)
    assert set(cluster_union.tolist()) == set(kept_union.tolist())
    for c in classes:
        assert set(_as_int(model.kept_[c]).tolist()) <= set(np.flatnonzero(y == c).tolist())

    # (b) and (c)
    pooled = model.eps_scale * merge_scale(X, np.sort(kept_union), metric, params)
    for c in classes:
        K, Q, P = _as_int(model.kept_[c]), _as_int(model.core_[c]), _as_int(model.prototypes_[c])
        eps = model.radii_[c]
        assert len(P) >= 1, f"class {c!r} has no prototype"
        if len(K) == 0:
            members = np.flatnonzero(y == c)
            d = distances_to(X[members], X[members].mean(axis=0, keepdims=True), metric, params).ravel()
            tau = int(members[np.argmin(d)])
            np.testing.assert_array_equal(Q, [tau])
            np.testing.assert_array_equal(P, [tau])
            assert eps == 0.0
            continue
        assert set(Q.tolist()) <= set(P.tolist()) <= set(K.tolist())
        assert eps == (0.0 if len(K) == 1 else pooled)
        cover = distances_to(X[K], X[Q], metric, params).min(axis=1).max()
        assert cover <= eps, f"class {c!r}: covering radius {cover} > eps {eps}"
        DQQ = distances_to(X[Q], X[Q], metric, params)
        off_diagonal = DQQ[~np.eye(len(Q), dtype=bool)]
        assert np.all(off_diagonal > eps), f"class {c!r}: two core points within eps of each other"


def assert_partition(Z, counts, purity, s_min, class_totals):
    """Assert that `accepted_leaf_assignment` maps every leaf to its highest accepted ancestor.

    Every leaf gets an id >= 0; every id is a node that passes the acceptance test (purity or size); the leaves
    assigned to an id are exactly the descendant leaves of that node; and no proper ancestor of an assigned
    node is itself acceptable.
    """
    n_nodes = len(counts)
    n = (n_nodes + 1) // 2
    assign = accepted_leaf_assignment(Z, counts, purity, s_min, class_totals)
    accept = node_purity_at_least(counts, purity, class_totals) | (counts.sum(axis=1) <= s_min)
    left, right = Z[:, 0].astype(int), Z[:, 1].astype(int)
    leaves = [{i} for i in range(n)]
    parent = np.full(n_nodes, -1, dtype=int)
    for j in range(n - 1):
        leaves.append(leaves[left[j]] | leaves[right[j]])
        parent[left[j]] = parent[right[j]] = n + j
    assert assign.shape == (n,)
    assert np.all(assign >= 0)
    for cid in np.unique(assign):
        assert accept[cid]
        assert set(np.flatnonzero(assign == cid).tolist()) == leaves[cid]
        a = parent[cid]
        while a != -1:
            assert not accept[a], f"node {cid} is assigned although its ancestor {a} is acceptable"
            a = parent[a]


# --------------------------------------------------------------------------------------------------------------
# Fixtures: the edge cases of Lemma 2 (repeated feature vectors, coincident rows of different classes, a pooled
# radius of zero, cross-class distance ties on an integer lattice, singleton and empty kept sets, degenerate sample
# sizes). Every fixture is run through assert_lemma2 under every completion and pruning option and both weightings.
# --------------------------------------------------------------------------------------------------------------


def _blobs(centers, n_per, rng, scale=0.6):
    X = np.vstack([rng.normal(c, scale, size=(n_per, len(c))) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_per)
    return X, y


def _noisy3():
    rng = np.random.default_rng(0)
    X, y = _blobs([(0.0, 0.0), (3.0, 0.0), (1.5, 2.5)], 30, rng, scale=0.8)
    flip = rng.random(len(y)) < 0.1
    return X, np.where(flip, (y + 1) % 3, y)


def _f1_same_class_duplicates():
    # Every fourth row of two blobs appears three times. Only a third of the merge heights of K are 0, so the
    # pooled radius stays positive (F4 is the zero-radius case).
    X, y = _blobs([(0.0, 0.0), (2.5, 0.5)], 30, np.random.default_rng(1))
    reps = np.where(np.arange(len(y)) % 4 == 0, 3, 1)
    return np.repeat(X, reps, axis=0), np.repeat(y, reps)


F2_SPOT = np.array([0.0, 0.0])


def _f2_small_mixed_coincident():
    # Three coincident rows (labels 0, 0, 1) at F2_SPOT, the centre of the class-0 blob; the group has size <= s_min.
    X, y = _blobs([(0.0, 0.0), (3.0, 0.0)], 30, np.random.default_rng(2))
    return np.vstack([X, np.tile(F2_SPOT, (3, 1))]), np.concatenate([y, [0, 0, 1]])


def _f3_large_mixed_coincident():
    # Six coincident rows at the origin, labels 0, 0, 0, 1, 1, 1 (rows 0..5), with a far blob per class. The
    # group has size > s_min and purity 1/2, so the descent must split it.
    X, y = _blobs([(10.0, 0.0), (-10.0, 0.0)], 20, np.random.default_rng(3))
    return np.vstack([np.zeros((6, 2)), X]), np.concatenate([[0, 0, 0, 1, 1, 1], y])


def _f4_eps_zero():
    # Every row appears three times, so two thirds of the merge heights of K are 0 and the pooled radius is 0.
    X, y = _blobs([(0.0, 0.0), (2.5, 0.5)], 12, np.random.default_rng(4))
    return np.repeat(X, 3, axis=0), np.repeat(y, 3)


def _f5_lattice_checkerboard():
    ij = np.array([(i, j) for i in range(5) for j in range(5)], dtype=float)
    return ij, (ij.sum(axis=1) % 2).astype(int)


def _f5_lattice_random():
    ij = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    return ij, np.random.default_rng(5).integers(0, 2, len(ij))


def _f6_singleton_class():
    # Class 1 is one isolated far row (row 30). With balanced weights the root has purity 1/2, so the row is
    # kept as its own cluster; counting points, the root is accepted and the row becomes a guard.
    X, y = _blobs([(0.0, 0.0)], 30, np.random.default_rng(6))
    return np.vstack([X, [[10.0, 10.0]]]), np.concatenate([y, [1]])


def _f7_guarded_multirow_class():
    # The fixture of test_balanced_purity_keeps_a_minority_cluster: with class_weight=None the root is accepted
    # and class 1 (10 rows) is emptied, so its guard is the centroid-nearest row of X_1.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (200, 2)), rng.normal((2.5, 0.0), 0.05, (10, 2))])
    return X, np.repeat([0, 1], [200, 10])


def _f10_pair(delta):
    # Class 1 is two far rows (30, 31) at distance delta; with balanced weights both are kept.
    X, y = _blobs([(0.0, 0.0)], 30, np.random.default_rng(10))
    return np.vstack([X, [[20.0, 0.0], [20.0 + delta, 0.0]]]), np.concatenate([y, [1, 1]])


def _one_class():
    return np.random.default_rng(9).normal(size=(20, 2)), np.zeros(20, dtype=int)


FIXTURES = {
    "noisy3": _noisy3,
    "F1_same_class_duplicates": _f1_same_class_duplicates,
    "F2_small_mixed_coincident": _f2_small_mixed_coincident,
    "F3_large_mixed_coincident": _f3_large_mixed_coincident,
    "F4_eps_zero": _f4_eps_zero,
    "F5_lattice_checkerboard": _f5_lattice_checkerboard,
    "F5_lattice_random": _f5_lattice_random,
    "F6_singleton_class": _f6_singleton_class,
    "F7_guarded_multirow_class": _f7_guarded_multirow_class,
    "F9_one_class": _one_class,
    "F10_pair_near": lambda: _f10_pair(0.05),
    "F10_pair_far": lambda: _f10_pair(3.0),
}

CONFIGS = [
    dict(completion=None),
    dict(completion="consistent", prune=False),
    dict(completion="consistent", prune=True),  # default
    dict(completion="gain", prune=False),
    dict(completion="gain", prune=True),
]
WEIGHTS = ["balanced", None]
CITYBLOCK_FIXTURES = ["F1_same_class_duplicates", "F5_lattice_checkerboard", "F6_singleton_class"]


def _config_id(cfg):
    return "_".join(f"{k}={v}" for k, v in cfg.items())


# --------------------------------------------------------------------------------------------------------------
# Lemma 2 on every fixture x config x weight
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("class_weight", WEIGHTS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("name", list(FIXTURES))
def test_lemma2(name, cfg, class_weight):
    X, y = FIXTURES[name]()
    assert_lemma2(PENS(class_weight=class_weight, **cfg).fit(X, y), X, y)


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
@pytest.mark.parametrize("name", CITYBLOCK_FIXTURES)
def test_lemma2_cityblock(name, cfg):
    X, y = FIXTURES[name]()
    assert_lemma2(PENS(metric="cityblock", **cfg).fit(X, y), X, y)


@pytest.mark.parametrize("s_min", [1, 5])
@pytest.mark.parametrize("purity", [0.7, 0.9, 1.0])
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("name", ["noisy3", "F3_large_mixed_coincident", "F5_lattice_checkerboard"])
def test_accepted_leaf_assignment_is_a_partition(name, balanced, purity, s_min):
    X, y = FIXTURES[name]()
    _, y_enc = np.unique(y, return_inverse=True)
    n_classes = y_enc.max() + 1
    totals = np.bincount(y_enc, minlength=n_classes)
    Z = single_linkage(X, "euclidean", {})
    counts = node_class_counts(Z, y_enc, n_classes)
    assert_partition(Z, counts, purity, s_min, totals if balanced else None)


# --------------------------------------------------------------------------------------------------------------
# Fixture-specific assertions
# --------------------------------------------------------------------------------------------------------------


def _kept_union(model):
    return np.concatenate([_as_int(k) for k in model.kept_.values()])


def _pooled_radius(model):
    radii = {model.radii_[c] for c in model.classes_ if len(model.kept_[c]) >= 2}
    assert len(radii) == 1
    return radii.pop()


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f1_core_has_no_coincident_rows_and_positive_radius(cfg):
    X, y = _f1_same_class_duplicates()
    model = PENS(**cfg).fit(X, y)
    assert _pooled_radius(model) > 0.0
    for c in model.classes_:
        Q = _as_int(model.core_[c])
        assert len(np.unique(X[Q], axis=0)) == len(Q)


@pytest.mark.parametrize("class_weight", WEIGHTS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f2_coincident_spot_carries_one_label_in_k(cfg, class_weight):
    X, y = _f2_small_mixed_coincident()
    model = PENS(class_weight=class_weight, **cfg).fit(X, y)
    at_spot = np.flatnonzero(np.all(X == F2_SPOT, axis=1))
    kept_at_spot = np.intersect1d(_kept_union(model), at_spot)
    assert len(kept_at_spot) >= 1
    assert set(y[kept_at_spot].tolist()) == {0}


def test_f3_record_mixed_label_coincidence_at_defaults(record_property):
    # Lemma 2 holds here (test_lemma2). Whether K keeps coincident rows of both labels is a Stage-1 fact that
    # the Remark 1 discussion (tie rules, Part 3 of the handoff) depends on, so it is recorded, not asserted.
    X, y = _f3_large_mixed_coincident()
    model = PENS().fit(X, y)
    group = np.arange(6)
    kept0 = np.intersect1d(_as_int(model.kept_[0]), group).tolist()
    kept1 = np.intersect1d(_as_int(model.kept_[1]), group).tolist()
    record_property("f3_kept_rows_at_coincident_location", f"class0={kept0} class1={kept1}")
    record_property("f3_k_has_coincident_rows_of_both_labels", bool(kept0 and kept1))
    assert set(kept0).isdisjoint(kept1)


@pytest.mark.parametrize("class_weight", WEIGHTS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f4_zero_radius_core_is_one_row_per_distinct_coordinate(cfg, class_weight):
    X, y = _f4_eps_zero()
    model = PENS(class_weight=class_weight, **cfg).fit(X, y)
    assert _pooled_radius(model) == 0.0
    for c in model.classes_:
        K = np.sort(_as_int(model.kept_[c]))
        if len(K) < 2:
            continue
        _, first = np.unique(X[K], axis=0, return_index=True)
        expected = np.sort(K[first])  # the smallest row index of every coincident group of K_c
        np.testing.assert_array_equal(model.core_[c], expected)
        D = distances_to(X[model.core_[c]], X[model.core_[c]], model.metric, model.metric_params)
        assert np.all(D[~np.eye(len(expected), dtype=bool)] > 0.0)


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f6_singleton_kept_set(cfg):
    X, y = _f6_singleton_class()
    model = PENS(**cfg).fit(X, y)
    assert model.radii_[1] == 0.0
    for attr in (model.kept_, model.core_, model.prototypes_):
        np.testing.assert_array_equal(attr[1], [30])


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f7_guard_of_a_multirow_class(cfg):
    X, y = _f7_guarded_multirow_class()
    model = PENS(class_weight=None, **cfg).fit(X, y)
    assert len(model.kept_[1]) == 0
    assert model.radii_[1] == 0.0
    X1 = X[200:]
    tau = 200 + int(np.argmin(np.linalg.norm(X1 - X1.mean(axis=0), axis=1)))
    np.testing.assert_array_equal(model.prototypes_[1], [tau])
    np.testing.assert_array_equal(model.core_[1], [tau])
    assert set(edit_cost(X, y, model.kept_)) == {0}


def test_f7_guarded_class_absent_from_certificate():
    pytest.importorskip("ripser")
    pytest.importorskip("gudhi")
    from pens import verify_certificate

    X, y = _f7_guarded_multirow_class()
    model = PENS(class_weight=None).fit(X, y)
    assert set(verify_certificate(X, model.kept_, model.prototypes_, model.radii_, maxdim=1)) == {0}


@pytest.mark.parametrize("class_weight", WEIGHTS)
@pytest.mark.parametrize("name", ["noisy3", "F5_lattice_checkerboard", "F3_large_mixed_coincident"])
def test_f8_s_min_at_least_n_keeps_one_whole_class(name, class_weight):
    X, y = FIXTURES[name]()
    model = PENS(s_min=len(y), class_weight=class_weight).fit(X, y)
    assert_lemma2(model, X, y)
    whole = [c for c in model.classes_ if set(model.kept_[c].tolist()) == set(np.flatnonzero(y == c).tolist())]
    assert len(whole) == 1
    for c in model.classes_:
        if c != whole[0]:
            assert len(model.kept_[c]) == 0
            assert len(model.prototypes_[c]) == 1


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f9_n_equals_one(cfg):
    model = PENS(**cfg).fit(np.array([[1.0, 2.0]]), np.array([7]))
    for attr in (model.kept_, model.core_, model.prototypes_):
        np.testing.assert_array_equal(attr[7], [0])
    assert model.radii_[7] == 0.0
    np.testing.assert_array_equal(model.prototype_indices_, [0])


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f9_one_class_keeps_every_row(cfg):
    X, y = _one_class()
    model = PENS(**cfg).fit(X, y)
    assert_lemma2(model, X, y)
    np.testing.assert_array_equal(np.sort(model.kept_[0]), np.arange(len(y)))


@pytest.mark.parametrize("cfg", CONFIGS, ids=_config_id)
def test_f10_two_row_kept_set(cfg):
    sizes = set()
    for delta in (0.05, 3.0):
        X, y = _f10_pair(delta)
        model = PENS(**cfg).fit(X, y)
        np.testing.assert_array_equal(np.sort(model.kept_[1]), [30, 31])
        eps = model.radii_[1]
        assert eps == model.radii_[0] == _pooled_radius(model)
        assert len(model.core_[1]) == (1 if delta <= eps else 2)
        sizes.add(len(model.core_[1]))
    assert sizes == {1, 2}


# --------------------------------------------------------------------------------------------------------------
# farthest_point_cover: strict stopping rule and tie rules (Section 4.2)
# --------------------------------------------------------------------------------------------------------------

_LINE = np.array([[0.0], [1.0], [2.0]])


def test_n1_distance_exactly_eps_is_covered_not_added():
    assert farthest_point_cover(_LINE, np.arange(3), 1.0, "euclidean", {}) == [1]


def test_n2_just_below_eps_adds_farthest_with_smaller_index_first():
    assert farthest_point_cover(_LINE, np.arange(3), 0.5, "euclidean", {}) == [1, 0, 2]


def test_n3_centroid_tie_seeds_smaller_index():
    X = np.array([[-1.0], [1.0]])
    assert farthest_point_cover(X, np.arange(2), np.inf, "euclidean", {}) == [0]
    assert farthest_point_cover(X, np.arange(2), 0.5, "euclidean", {}) == [0, 1]


@pytest.mark.parametrize("eps", [0.3, 0.8])
def test_n4_core_points_are_strictly_separated(eps):
    X = np.random.default_rng(0).normal(size=(200, 2))
    members = np.arange(200)
    Q = farthest_point_cover(X, members, eps, "euclidean", {})
    D = distances_to(X[Q], X[Q], "euclidean", {})
    assert np.all(D[~np.eye(len(Q), dtype=bool)] > eps)
    assert distances_to(X[members], X[Q], "euclidean", {}).min(axis=1).max() <= eps


def test_n5_eps_zero_with_duplicates_picks_smallest_index_per_group():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [2.0]])
    assert farthest_point_cover(X, np.arange(5), 0.0, "euclidean", {}) == [2, 0, 4]


def test_n6_ties_follow_member_order_not_row_index():
    # Rows 3 and 7 are equidistant from the seed, row 12. Sorted members break the tie towards the smaller row
    # index, as Section 4.2 states; unsorted members break it towards the earlier member.
    X = np.zeros((13, 1))
    X[3], X[7], X[12] = -1.0, 1.0, 0.0
    assert farthest_point_cover(X, np.array([3, 7, 12]), 0.0, "euclidean", {}) == [12, 3, 7]
    assert farthest_point_cover(X, np.array([7, 3, 12]), 0.0, "euclidean", {}) == [12, 7, 3]


def test_fit_passes_sorted_kept_rows_to_the_cover(monkeypatch):
    # The smaller-row-index rule of Section 4.2 holds only because fit sorts K_c before covering it (N6).
    seen = []
    original = selector.farthest_point_cover

    def spy(X, members, eps, metric, metric_params):
        seen.append(np.asarray(members).copy())
        return original(X, members, eps, metric, metric_params)

    monkeypatch.setattr(selector, "farthest_point_cover", spy)
    X, y = _noisy3()
    model = PENS().fit(X, y)
    assert len(seen) == len(model.classes_)
    for members in seen:
        assert np.all(np.diff(members) > 0)
    # The check is only meaningful if concatenating the clusters of some class gives an unsorted array.
    raw = [np.concatenate([_as_int(k) for cc, k in model.clusters_ if cc == c]) for c in model.classes_]
    assert any(np.any(np.diff(r) < 0) for r in raw)


# --------------------------------------------------------------------------------------------------------------
# Invariance: K and eps do not depend on completion; the core does not depend on completion or pruning
# --------------------------------------------------------------------------------------------------------------

_PENS_GRID = [
    dict(eps_scale=lam, completion=completion, prune=prune)
    for lam in (1.0, 2.0, 4.0, 8.0, 16.0)
    for completion in ("gain", "consistent")
    for prune in (True, False)
] + [dict(completion=None), dict(completion="gain", gain=0.0), dict(completion="gain", gain=3.0)]


def _assert_same_dict(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


@pytest.mark.parametrize("name", ["noisy3", "F5_lattice_checkerboard"])
def test_v1_kept_and_clusters_independent_of_completion_and_radius(name):
    X, y = FIXTURES[name]()
    ref = PENS().fit(X, y)
    for cfg in _PENS_GRID:
        model = PENS(**cfg).fit(X, y)
        _assert_same_dict(model.kept_, ref.kept_)
        assert len(model.clusters_) == len(ref.clusters_)
        for (la, ka), (lb, kb) in zip(model.clusters_, ref.clusters_):
            assert la == lb
            np.testing.assert_array_equal(ka, kb)


@pytest.mark.parametrize("eps_scale", [1.0, 4.0])
@pytest.mark.parametrize("name", ["noisy3", "F5_lattice_checkerboard"])
def test_v2_core_independent_of_completion_and_pruning(name, eps_scale):
    X, y = FIXTURES[name]()
    ref = PENS(eps_scale=eps_scale, completion=None).fit(X, y)
    for cfg in CONFIGS:
        model = PENS(eps_scale=eps_scale, **cfg).fit(X, y)
        _assert_same_dict(model.core_, ref.core_)
        assert model.radii_ == ref.radii_


# --------------------------------------------------------------------------------------------------------------
# Ripser and GUDHI conventions the certificate relies on (Lemma 1)
# --------------------------------------------------------------------------------------------------------------


def test_ripser_conventions():
    ripser = pytest.importorskip("ripser")
    gudhi = pytest.importorskip("gudhi")
    from scipy.spatial.distance import pdist, squareform

    def dgms(P, maxdim=1):
        return ripser.ripser(squareform(pdist(P)), distance_matrix=True, maxdim=maxdim)["dgms"]

    # 1. Edge-length (diameter) convention: two points at distance 1 die at 1 in H0.
    h0 = dgms(np.array([[0.0], [1.0]]), maxdim=0)[0]
    np.testing.assert_array_equal(h0, [[0.0, 1.0], [0.0, np.inf]])

    # 2. A unit square has one H1 bar [1, sqrt(2)]. Ripser computes in float32.
    h1 = dgms(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))[1]
    assert h1.shape == (1, 2)
    assert h1[0, 0] == 1.0
    assert h1[0, 1] == pytest.approx(np.sqrt(2.0), rel=1e-6)

    # 3. No zero-length bars: three coincident rows plus one other give only [0, 1] and [0, inf] in H0.
    h0 = dgms(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), maxdim=0)[0]
    np.testing.assert_array_equal(h0, [[0.0, 1.0], [0.0, np.inf]])

    # 4. Only the H0 bar (0, inf) is essential on the full filtration (thresh=inf).
    t = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False)
    circle = dgms(np.column_stack([np.cos(t), np.sin(t)]))
    assert [int(np.isinf(d[:, 1]).sum()) for d in circle] == [1, 0]
    assert len(circle[1]) == 1

    # 5. GUDHI matches an unmatched bar of length L to the diagonal at cost L / 2.
    assert gudhi.bottleneck_distance(np.array([[0.0, 2.0]]), np.empty((0, 2))) == 1.0

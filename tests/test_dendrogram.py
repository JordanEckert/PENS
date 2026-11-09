import numpy as np

from pens.dendrogram import (
    accepted_leaf_assignment,
    descend,
    edit_clusters,
    node_class_counts,
    node_purity_at_least,
    single_linkage,
)


def _data():
    """Build a small noisy two-class fixture.

    Returns:
        Tuple (X, y_enc, n_classes, class_totals).
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (20, 2)), rng.normal(2.0, 1.0, (20, 2))])
    y = np.repeat([0, 1], 20)
    y[rng.random(40) < 0.15] ^= 1
    return X, y, 2, np.bincount(y, minlength=2)


def test_counts_root_and_leaves():
    X, y, k, _ = _data()
    n = len(X)
    Z = single_linkage(X, "euclidean", {})
    assert Z.shape == (n - 1, 4)
    counts = node_class_counts(Z, y, k)
    assert counts.shape == (2 * n - 1, k)
    assert counts[-1].sum() == n
    np.testing.assert_array_equal(counts[-1], np.bincount(y, minlength=k))
    np.testing.assert_array_equal(counts[:n].sum(axis=1), np.ones(n))
    np.testing.assert_array_equal(counts[np.arange(n), y], np.ones(n))


def test_purity_one_gives_pure_clusters():
    X, y, k, _ = _data()
    Z = single_linkage(X, "euclidean", {})
    counts = node_class_counts(Z, y, k)
    leaf_assign = accepted_leaf_assignment(Z, counts, 1.0, 1)
    assert leaf_assign.shape == (len(X),)
    for cid in np.unique(leaf_assign):
        assert len(np.unique(y[leaf_assign == cid])) == 1


def test_large_s_min_gives_single_cluster():
    X, y, k, totals = _data()
    n = len(X)
    Z = single_linkage(X, "euclidean", {})
    counts = node_class_counts(Z, y, k)
    leaf_assign = accepted_leaf_assignment(Z, counts, 1.0, n)
    np.testing.assert_array_equal(leaf_assign, np.full(n, 2 * n - 2))
    clusters = descend(X, y, k, totals, 1.0, n, 1, "euclidean", {})
    assert len(clusters) == 1


def test_tie_break_prefers_rarer_class():
    leaf_assign = np.zeros(4, dtype=np.int64)
    y = np.array([0, 0, 1, 1])
    clusters = edit_clusters(leaf_assign, y, np.array([10, 3]), 1)
    assert len(clusters) == 1
    label, kept = clusters[0]
    assert label == 1
    np.testing.assert_array_equal(kept, [2, 3])
    label, _ = edit_clusters(leaf_assign, y, np.array([3, 3]), 1)[0]
    assert label == 0


def test_min_keep_drops_small_clusters():
    leaf_assign = np.array([5, 5, 5, 6])
    y = np.array([0, 0, 1, 1])
    clusters = edit_clusters(leaf_assign, y, np.array([2, 2]), 2)
    assert len(clusters) == 1
    assert clusters[0][0] == 0
    np.testing.assert_array_equal(clusters[0][1], [0, 1])


def test_node_purity_at_least():
    counts = np.array([[9, 1], [8, 1], [1, 1], [1, 2]])
    np.testing.assert_array_equal(node_purity_at_least(counts, 0.9), [True, False, False, False])
    # Weighted by 1 / (1, 9), the third node has purity 1 / (1 + 1/9) = 9/10, which is 0.8999999999999999 in floating
    # point, and the fourth 9/11.
    np.testing.assert_array_equal(node_purity_at_least(counts, 0.9, np.array([1, 9])), [True, True, True, False])
    np.testing.assert_array_equal(node_purity_at_least(counts, 1.0, np.array([1, 9])), [False, False, False, False])


def test_node_of_purity_exactly_purity_is_accepted():
    # Class 0 is one point, with one point of class 1 next to it and the other eight far away. With balanced weights
    # the pair has purity 9/10, so at purity=0.9 it is accepted and its class-1 point is removed.
    X = np.concatenate([[0.0, 1.0], np.arange(100.0, 108.0)])[:, None]
    y = np.array([0] + [1] * 9)
    clusters = descend(X, y, 2, np.bincount(y), 0.9, 1, 1, "euclidean", {}, balanced=True)
    kept = np.sort(np.concatenate([m for _, m in clusters]))
    np.testing.assert_array_equal(kept, [0, *range(2, 10)])


def test_balanced_majority_goes_to_the_denser_class():
    leaf_assign = np.zeros(4, dtype=np.int64)
    y = np.array([0, 0, 0, 1])
    label, kept = edit_clusters(leaf_assign, y, np.array([300, 10]), 1, balanced=True)[0]
    assert label == 1
    np.testing.assert_array_equal(kept, [3])
    label, _ = edit_clusters(leaf_assign, y, np.array([300, 10]), 1, balanced=False)[0]
    assert label == 0

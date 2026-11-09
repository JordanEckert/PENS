import numpy as np
import pytest

from pens.completion import balanced_weights, consistent_completion, gain_completion, prune_completion


def _nearest_labels(X, y, T, S):
    """Label of the nearest prototype of every target point (first prototype on ties)."""
    S = np.asarray(S)
    d = np.linalg.norm(X[T][:, None, :] - X[S][None, :, :], axis=2)
    return y[S[np.argmin(d, axis=1)]]


def _two_blobs():
    """Two separated blobs, a misclassified group of five points and one isolated misclassified point.

    Returns:
        Tuple (X, y, T, S0) where S0 holds one point per blob.
    """
    rng = np.random.default_rng(0)
    a = rng.normal((0.0, 0.0), 0.3, (40, 2))
    b = rng.normal((4.0, 0.0), 0.3, (40, 2))
    group = rng.normal((1.6, 2.0), 0.1, (5, 2))  # class 1, but nearer the class-0 prototype
    lone = np.array([[0.2, 0.1]])  # class 1, inside the class-0 blob
    X = np.vstack([a, b, group, lone])
    y = np.array([0] * 40 + [1] * 40 + [1] * 5 + [1])
    S0 = [0, 40]
    X[0] = (0.0, 0.0)
    X[40] = (4.0, 0.0)
    return X, y, np.arange(len(X)), S0


def test_balanced_weights():
    w = balanced_weights(np.array([0, 0, 0, 1]), 3)
    assert w.mean() == pytest.approx(1.0)
    assert w[:3].sum() == pytest.approx(w[3])


def test_consistent_completion_is_consistent():
    X, y, T, S0 = _two_blobs()
    S = consistent_completion(X, y, T, S0, "euclidean", {})
    assert S[:2] == S0
    assert np.all(_nearest_labels(X, y, T, S) == y[T])


def test_gain_completion_skips_isolated_point():
    X, y, T, S0 = _two_blobs()
    S = gain_completion(X, y, 2, T, S0, 1.0, np.ones(len(T)), "euclidean", {})
    added = set(S) - set(S0)
    assert len(added & set(range(80, 85))) == 1  # one representative corrects the whole group
    assert 85 not in added  # the isolated point can only correct itself
    assert np.all(_nearest_labels(X, y, np.arange(80, 85), S) == 1)


def test_prune_keeps_protected_and_accuracy():
    X, y, T, S0 = _two_blobs()
    S = consistent_completion(X, y, T, S0, "euclidean", {})
    w = np.ones(len(T))
    before = (_nearest_labels(X, y, T, S) == y[T]).sum()
    kept = prune_completion(X, y, T, S + [5, 6, 45], set(S0), w, "euclidean", {})
    assert set(S0) <= set(kept)
    assert kept == sorted(kept)
    assert (_nearest_labels(X, y, T, kept) == y[T]).sum() >= before
    assert len(kept) < len(set(S + [5, 6, 45]))

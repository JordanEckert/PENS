import numpy as np
import pytest

from pens._metrics import distances_to
from pens.net import centroid_nearest, farthest_point_cover, merge_scale


def _points():
    """Draw a small 2-D point cloud.

    Returns:
        (50, 2) float array.
    """
    return np.random.default_rng(0).normal(size=(50, 2))


@pytest.mark.parametrize("metric", ["euclidean", "cityblock"])
@pytest.mark.parametrize("eps", [0.3, 0.8, 1.5])
def test_cover_radius_within_eps(metric, eps):
    X = _points()
    members = np.arange(5, 45)
    protos = farthest_point_cover(X, members, eps, metric, {})
    assert set(protos) <= set(members.tolist())
    d = distances_to(X[members], X[protos], metric, {})
    assert d.min(axis=1).max() <= eps


def test_single_member_returns_itself():
    X = _points()
    assert farthest_point_cover(X, np.array([17]), 0.0, "euclidean", {}) == [17]


def test_infinite_eps_returns_seed():
    X = _points()
    members = np.arange(10, 30)
    seed = members[centroid_nearest(X[members], "euclidean", {})]
    assert farthest_point_cover(X, members, np.inf, "euclidean", {}) == [int(seed)]


def test_merge_scale_small_and_positive():
    X = _points()
    assert merge_scale(X, np.array([3]), "euclidean", {}) == 0.0
    assert merge_scale(X, np.array([], dtype=int), "euclidean", {}) == 0.0
    assert merge_scale(X, np.arange(20), "euclidean", {}) > 0.0

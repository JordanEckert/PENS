import numpy as np
import pytest

from pens import PENS, bottleneck_bound, edit_cost, end_to_end_bound, hausdorff_distance, verify_certificate


def _data():
    """Build a small two-class fixture with ring structure in one class.

    Returns:
        Tuple (X, y).
    """
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 2.0 * np.pi, 40)
    ring = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0.0, 0.05, (40, 2))
    blob = rng.normal((3.0, 0.0), 0.4, (30, 2))
    return np.vstack([ring, blob]), np.repeat([0, 1], [40, 30])


def _noisy_data():
    """The fixture of _data with five labels flipped, so that the edit removes points.

    Returns:
        Tuple (X, y).
    """
    X, y = _data()
    y = y.copy()
    y[[3, 11, 19, 45, 52]] ^= 1
    return X, y


SELECTORS = [
    lambda e: PENS(eps_scale=e),
    lambda e: PENS(eps_scale=e, completion="gain"),
    lambda e: PENS(eps_scale=e, completion=None),
    lambda e: PENS(eps_scale=e, class_weight=None, min_keep=2, completion=None),
]


@pytest.mark.parametrize("make", SELECTORS)
@pytest.mark.parametrize("eps_scale", [1.0, 2.0, 4.0])
def test_certificate_holds(eps_scale, make):
    pytest.importorskip("ripser")
    pytest.importorskip("gudhi")
    X, y = _data()
    model = make(eps_scale).fit(X, y)
    res = verify_certificate(X, model.kept_, model.prototypes_, model.radii_, maxdim=1)
    assert len(res) > 0
    for label, r in res.items():
        assert r["eps"] == pytest.approx(model.radii_[label])
        assert r["bound"] == pytest.approx(2.0 * r["eps"])
        assert set(r["bottleneck"]) == {0, 1}
        assert r["hausdorff"] <= r["eps"] + 1e-9
        assert bool(r["holds"])


def test_hausdorff_symmetric_and_zero():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(15, 3))
    B = rng.normal(size=(9, 3))
    assert hausdorff_distance(A, A) == 0.0
    assert hausdorff_distance(A, B) == pytest.approx(hausdorff_distance(B, A))
    assert hausdorff_distance(A, B, metric="cityblock") > 0.0


def test_bottleneck_bound_doubles():
    assert bottleneck_bound({0: 1.5, "a": 0.0}) == {0: 3.0, "a": 0.0}


def test_edit_cost_and_end_to_end_bound():
    X, y = _noisy_data()
    model = PENS(eps_scale=2.0).fit(X, y)
    cost = edit_cost(X, y, model.kept_)
    bound = end_to_end_bound(X, y, model.kept_, model.radii_)
    assert set(cost) == {c for c in model.classes_ if len(model.kept_[c])}
    for label, c in cost.items():
        Xc = X[y == label]
        assert c == pytest.approx(hausdorff_distance(X[model.kept_[label]], Xc))
        assert bound[label] == pytest.approx(2.0 * (model.radii_[label] + c))
    assert max(cost.values()) > 0.0
    clean = PENS(eps_scale=2.0).fit(*_data())
    X0, y0 = _data()
    assert all(v == 0.0 for k, v in edit_cost(X0, y0, clean.kept_).items() if len(clean.kept_[k]) == (y0 == k).sum())


def test_end_to_end_bound_holds():
    ripser = pytest.importorskip("ripser")
    gudhi = pytest.importorskip("gudhi")
    from scipy.spatial.distance import pdist, squareform

    def dgms(P):
        d = ripser.ripser(squareform(pdist(P)), distance_matrix=True, maxdim=1)["dgms"]
        return [g[np.isfinite(g[:, 1])] for g in d]

    X, y = _noisy_data()
    for make in SELECTORS:
        model = make(2.0).fit(X, y)
        bound = end_to_end_bound(X, y, model.kept_, model.radii_)
        for label, b in bound.items():
            P = model.prototypes_[label]
            if len(P) < 2:
                continue
            dp, dx = dgms(X[P]), dgms(X[y == label])
            for k in (0, 1):
                assert gudhi.bottleneck_distance(dp[k], dx[k]) <= b + 1e-9

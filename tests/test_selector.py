import numpy as np
import pytest

from pens import PENS

SETTINGS = [
    dict(purity=0.9, eps_scale=2.0, s_min=5, min_keep=2),
    dict(purity=1.0, eps_scale=0.5, s_min=1, min_keep=1),
    dict(purity=0.7, eps_scale=4.0, s_min=10, min_keep=3),
]


def _blobs(centers, n_per, rng, scale=0.6):
    """Draw isotropic Gaussian blobs of n_per points around each center with labels 0..k-1."""
    X = np.vstack([rng.normal(c, scale, size=(n_per, len(c))) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_per)
    return X, y


def _fixtures():
    """Build the equivalence fixtures as (name, X, y, metric) tuples."""
    rng = np.random.default_rng(0)
    X2, y2 = _blobs([(0.0, 0.0), (2.5, 0.5)], 40, rng)
    X3, y3 = _blobs([(0.0, 0.0), (3.0, 0.0), (1.5, 2.5)], 30, rng, scale=0.8)
    flip = rng.random(len(y3)) < 0.1
    y3 = np.where(flip, (y3 + 1) % 3, y3)
    y3 = np.array(["a", "b", "c"])[y3]
    Xm, ym = _blobs([(0.0, 0.0, 0.0), (2.0, 2.0, 0.0)], 25, rng, scale=1.0)
    return [
        ("blobs2", X2, y2, "euclidean"),
        ("noisy3", X3, y3, "euclidean"),
        ("manhattan", Xm, ym, "manhattan"),
        ("n0", np.empty((0, 2)), np.empty(0, dtype=int), "euclidean"),
        ("n1", np.array([[1.0, 2.0]]), np.array([7]), "euclidean"),
        ("n2", np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]), "euclidean"),
    ]


def _assert_same(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_deterministic():
    _, X, y, _ = _fixtures()[1]
    first = PENS().fit(X, y).prototypes_
    second = PENS().fit(X, y).prototypes_
    _assert_same(first, second)


@pytest.mark.parametrize("params", SETTINGS)
def test_every_class_has_prototype(params):
    _, X, y, _ = _fixtures()[1]
    model = PENS(**params).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    for label in model.classes_:
        assert len(model.prototypes_[label]) >= 1


@pytest.mark.parametrize(
    "bad",
    [dict(purity=0.0), dict(purity=1.5), dict(eps_scale=0.0), dict(s_min=0), dict(min_keep=0)],
)
def test_bad_hyperparameters(bad):
    _, X, y, _ = _fixtures()[0]
    with pytest.raises(ValueError):
        PENS(**bad).fit(X, y)


def test_transform_and_flat_indices():
    _, X, y, _ = _fixtures()[0]
    model = PENS().fit(X, y)
    union = np.sort(np.concatenate(list(model.prototypes_.values())))
    np.testing.assert_array_equal(model.prototype_indices_, union)
    Xp, yp = model.transform(X, y)
    assert Xp.shape == (len(union), X.shape[1])
    assert yp.shape == (len(union),)
    np.testing.assert_array_equal(yp, y[union])


def test_guarded_classes():
    model = PENS(min_keep=2).fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    for label in (0, 1):
        np.testing.assert_array_equal(model.prototypes_[label], [label])
        assert len(model.kept_[label]) == 0
        assert model.radii_[label] == 0.0


def test_guarded_class_with_defaults():
    # The two-point root is accepted by size; the count tie goes to class 0 and class 1 is guarded.
    model = PENS().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    np.testing.assert_array_equal(model.kept_[0], [0])
    assert len(model.kept_[1]) == 0
    for label in (0, 1):
        np.testing.assert_array_equal(model.prototypes_[label], [label])
        assert model.radii_[label] == 0.0


@pytest.mark.parametrize(
    "bad",
    [dict(class_weight="auto"), dict(completion="cnn"), dict(gain=-1.0)],
)
def test_bad_new_hyperparameters(bad):
    with pytest.raises(ValueError):
        PENS(**bad)


CONFIGS = [
    dict(),
    dict(completion="gain"),
    dict(completion="gain", prune=False),
    dict(completion="consistent", prune=False),
    dict(completion=None),
    dict(class_weight=None, min_keep=2),
]


@pytest.mark.parametrize("params", CONFIGS)
def test_core_covers_kept_points(params):
    _, X, y, _ = _fixtures()[1]
    model = PENS(**params).fit(X, y)
    for label in model.classes_:
        kept = np.asarray(model.kept_[label])
        core = set(model.core_[label].tolist())
        protos = set(model.prototypes_[label].tolist())
        assert core <= protos
        if len(kept) == 0:
            assert len(protos) == 1
            continue
        assert protos <= set(kept.tolist())
        d = np.linalg.norm(X[kept][:, None, :] - X[model.core_[label]][None, :, :], axis=2)
        assert d.min(axis=1).max() <= model.radii_[label] + 1e-12


@pytest.mark.parametrize("prune", [True, False])
def test_consistent_completion_classifies_every_kept_point(prune):
    _, X, y, _ = _fixtures()[1]
    model = PENS(completion="consistent", prune=prune, eps_scale=8.0).fit(X, y)
    kept = np.concatenate([np.asarray(model.kept_[c]) for c in model.classes_])
    P = model.prototype_indices_
    d = np.linalg.norm(X[kept][:, None, :] - X[P][None, :, :], axis=2)
    same = np.asarray(y)[P][None, :] == np.asarray(y)[kept][:, None]
    own = np.where(same, d, np.inf).min(axis=1)
    other = np.where(~same, d, np.inf).min(axis=1)
    assert np.all(own <= other)


def test_global_radius_is_shared():
    _, X, y, _ = _fixtures()[1]
    model = PENS().fit(X, y)
    radii = {model.radii_[c] for c in model.classes_ if len(model.kept_[c]) >= 2}
    assert len(radii) == 1


def test_balanced_purity_keeps_a_minority_cluster():
    # A rare class sits in a tight cluster at the edge of a large class. Counting points, the root is 95%
    # pure and is accepted, so the rare class is removed; with balanced weights the root is only 50% pure.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (200, 2)), rng.normal((2.5, 0.0), 0.05, (10, 2))])
    y = np.repeat([0, 1], [200, 10])
    counted = PENS(class_weight=None, completion=None).fit(X, y)
    balanced = PENS(class_weight="balanced", completion=None).fit(X, y)
    assert len(counted.kept_[1]) == 0
    assert len(balanced.kept_[1]) == 10


def test_min_keep_one_keeps_peeled_singletons():
    # In 10 dimensions the single-linkage dendrogram chains, and the descent peels off single points.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 10))
    y = np.repeat([0, 1], 60)
    X[60:, 0] += 1.5
    one = PENS(class_weight=None, min_keep=1, completion=None).fit(X, y)
    two = PENS(class_weight=None, min_keep=2, completion=None).fit(X, y)
    k1 = set(np.concatenate([np.asarray(v) for v in one.kept_.values()]).tolist())
    k2 = set(np.concatenate([np.asarray(v) for v in two.kept_.values()]).tolist())
    singles = {int(m[0]) for _, m in one.clusters_ if len(m) == 1}
    assert k2 <= k1
    assert k1 - k2 == singles
    assert len(singles) > 0


def test_select_and_fitted_attributes():
    _, X, y, _ = _fixtures()[1]
    model = PENS()
    _assert_same(model.select(X, y), model.prototypes_)
    for label, kept in model.clusters_:
        assert label in model.classes_
        assert np.all(np.asarray(y)[kept] == label)
    for label in model.classes_:
        assert model.radii_[label] >= 0.0
        assert np.all(np.asarray(y)[model.kept_[label]] == label)

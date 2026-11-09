import sys
from pathlib import Path

import numpy as np
import pytest

from pens import PENS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic"))
from run_remark1 import continuous_dataset, equidistant, failures, tie_heavy_dataset  # noqa: E402

from tests.test_invariants import _f3_large_mixed_coincident  # noqa: E402

# Remark 1 on about 200 seeded datasets of each kind (the full-size experiments are benchmarks/synthetic/run_remark1.py,
# which also defines the datasets, the failure test and the equidistance test used here).
SEEDS = range(200)


@pytest.mark.parametrize("prune", [False, True])
def test_tie_heavy_datasets(prune):
    for seed in SEEDS:
        X, y, cfg = tie_heavy_dataset(seed)
        m = PENS(prune=prune, **cfg).fit(X, y)
        f = failures(X, y, m)
        if not prune:
            assert len(f) == 0, (seed, f)
        elif len(f):
            # Every failure under pruning sits in a dataset with a kept point equidistant from two labels of K u G.
            assert len(equidistant(X, y, m)) > 0, (seed, f)


@pytest.mark.parametrize("prune", [False, True])
def test_continuous_datasets(prune):
    for seed in SEEDS:
        X, y, cfg = continuous_dataset(seed)
        assert len(failures(X, y, PENS(prune=prune, **cfg).fit(X, y))) == 0, seed


def test_coincident_rows_of_both_labels_are_kept():
    # Six coincident rows labelled 0, 0, 0, 1, 1, 1 (rows 0-5) with a far pure blob per class: at the defaults the
    # edit keeps rows 0, 1 and 2 for class 0 and row 5 for class 1, so K holds coincident rows of both labels and
    # the distance-0 case of Remark 1 occurs.
    X, y = _f3_large_mixed_coincident()
    m = PENS().fit(X, y)
    np.testing.assert_array_equal(np.intersect1d(m.kept_[0], np.arange(6)), [0, 1, 2])
    np.testing.assert_array_equal(np.intersect1d(m.kept_[1], np.arange(6)), [5])
    assert set(equidistant(X, y, m).tolist()) >= {0, 1, 2, 5}

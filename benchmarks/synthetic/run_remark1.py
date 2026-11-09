from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from pens import PENS

RESULTS = Path(__file__).parent / "results"
PURITIES, S_MINS, EPS_SCALES = (0.6, 0.8, 0.9, 1.0), (1, 2, 5), (0.5, 1.0, 4.0)
N_TIE, N_CONT = 400, 1000


def settings(rng):
    """One combination of purity, s_min and eps_scale, drawn uniformly from PURITIES x S_MINS x EPS_SCALES.

    The grid is {0.6, 0.8, 0.9, 1} x {1, 2, 5} x {0.5, 1, 4}; completion is always consistent.

    Args:
        rng: numpy Generator.

    Returns:
        Dict of PENS keyword arguments: purity, s_min, eps_scale and completion="consistent".
    """
    return dict(purity=float(rng.choice(PURITIES)), s_min=int(rng.choice(S_MINS)),
                eps_scale=float(rng.choice(EPS_SCALES)), completion="consistent")


def labels(rng, X, n_classes):
    """Random labels, or a deterministic pattern on the coordinates with up to 30% of the labels flipped.

    With probability 1/2 the labels are uniform over the classes; otherwise label = floor(x_1 + x_2) mod n_classes,
    each label then flipped to another class independently with a probability drawn from {0, 0.1, 0.2, 0.3}. The
    draw is repeated until at least two classes are present.

    Args:
        rng: numpy Generator.
        X: (n, 2) coordinates.
        n_classes: Number of classes.

    Returns:
        (n,) integer labels with at least two distinct values.
    """
    n = len(X)
    while True:
        if rng.random() < 0.5:
            y = rng.integers(0, n_classes, n)
        else:
            y = (np.floor(X[:, 0] + X[:, 1]) % n_classes).astype(int)
            flip = rng.random(n) < rng.choice([0.0, 0.1, 0.2, 0.3])
            y[flip] = (y[flip] + rng.integers(1, n_classes, int(flip.sum()))) % n_classes
        if len(np.unique(y)) >= 2:
            return y


def tie_heavy_dataset(seed):
    """A small dataset with many exact ties: rows on an integer grid, or continuous rows rounded to one decimal.

    n from 8 to 40 rows and 2 to 4 classes. With probability 1/2 the rows are integer points of the grid
    {0, ..., g - 1}^2 with g drawn from {3, 4, 5}, otherwise standard normal rows rounded to one decimal, so that
    duplicate rows and tied distances are common. Labels come from labels() and the PENS settings from settings().

    Args:
        seed: Dataset index; the rng is default_rng([seed, 1]).

    Returns:
        Tuple (X, y, cfg): (n, 2) float coordinates, integer labels, and the keyword arguments for PENS.
    """
    rng = np.random.default_rng([seed, 1])
    n, n_classes = int(rng.integers(8, 41)), int(rng.integers(2, 5))
    if rng.random() < 0.5:
        X = rng.integers(0, int(rng.choice([3, 4, 5])), (n, 2)).astype(float)
    else:
        X = np.round(rng.normal(size=(n, 2)), 1)
    return X, labels(rng, X, n_classes), settings(rng)


def continuous_dataset(seed):
    """A small dataset with standard normal coordinates and no duplicate rows.

    The same sizes, labels and settings as tie_heavy_dataset (n from 8 to 40, 2 to 4 classes); the absence of
    duplicate rows is asserted.

    Args:
        seed: Dataset index; the rng is default_rng([seed, 2]).

    Returns:
        Tuple (X, y, cfg) as in tie_heavy_dataset.
    """
    rng = np.random.default_rng([seed, 2])
    n, n_classes = int(rng.integers(8, 41)), int(rng.integers(2, 5))
    X = rng.normal(size=(n, 2))
    assert len(np.unique(X, axis=0)) == n
    return X, labels(rng, X, n_classes), settings(rng)


def kept_and_guards(model):
    """Row indices of K (all kept points) and of G (the prototypes of the classes whose kept set is empty).

    Args:
        model: Fitted PENS.

    Returns:
        Tuple (K, G) of integer index arrays; G is empty when the edit emptied no class.
    """
    K = np.concatenate([np.asarray(model.kept_[c], dtype=int) for c in model.classes_])
    G = np.concatenate([np.asarray(model.prototypes_[c], dtype=int) for c in model.classes_
                        if len(model.kept_[c]) == 0] + [np.empty(0, dtype=int)])
    return K, G


def failures(X, y, model):
    """Kept points with a prototype of another class strictly closer than every prototype of their own class.

    These are the failures of the Remark 1 experiment: 1-NN on the prototypes misclassifies such a kept point
    whatever its tie rule.

    Args:
        X: (n, d) coordinates.
        y: (n,) labels.
        model: Fitted PENS.

    Returns:
        Array of row indices of the failing kept points, possibly empty.
    """
    K, _ = kept_and_guards(model)
    P = np.asarray(model.prototype_indices_, dtype=int)
    D = cdist(X[K], X[P])
    same = y[P][None, :] == y[K][:, None]
    own = np.where(same, D, np.inf).min(axis=1)
    other = np.where(~same, D, np.inf).min(axis=1)
    return K[other < own]


def equidistant(X, y, model):
    """Kept points that have, at some single distance, points of K u G of at least two different labels.

    K is the set of all kept points and G the guard points of the classes the edit empties (kept_and_guards). A
    kept point coinciding with a point of another class is the distance-0 case. By Remark 1 such exact ties are the
    only source of failures when pruning is on.

    Args:
        X: (n, d) coordinates.
        y: (n,) labels.
        model: Fitted PENS.

    Returns:
        Array of row indices of the equidistant kept points, possibly empty.
    """
    K, G = kept_and_guards(model)
    KG = np.concatenate([K, G])
    D = cdist(X[K], X[KG])
    out = []
    for i, row in enumerate(D):
        for d in np.unique(row):
            if len(np.unique(y[KG][row == d])) >= 2:
                out.append(K[i])
                break
    return np.array(out, dtype=int)


def run(make, seeds):
    """Fit every dataset with pruning off and on and collect the datasets with failures.

    Args:
        make: Dataset constructor, tie_heavy_dataset or continuous_dataset.
        seeds: Iterable of dataset indices.

    Returns:
        Dict prune -> list of (seed, number of failing kept points, whether an equidistant kept point exists), one
        entry per dataset with at least one failure under that pruning setting.
    """
    out = {False: [], True: []}
    for seed in seeds:
        X, y, cfg = make(seed)
        for prune in (False, True):
            m = PENS(prune=prune, **cfg).fit(X, y)
            f = failures(X, y, m)
            if len(f):
                out[prune].append((seed, len(f), len(equidistant(X, y, m)) > 0))
    return out


def report(name, n, out):
    """Text lines of one experiment: failing points and datasets per pruning setting, then one line per dataset.

    Args:
        name: Experiment name ("Tie-heavy" or "Continuous").
        n: Number of datasets in the experiment.
        out: Dict prune -> list of (seed, number of failures, equidistant kept point present), from run().

    Returns:
        List of text lines.
    """
    lines = [f"{name}: {n} datasets, consistent completion."]
    for prune in (False, True):
        rows = out[prune]
        points, sets = sum(r[1] for r in rows), len(rows)
        lines.append(f"  pruning {'on' if prune else 'off'}: {points} failing kept points in {sets} datasets"
                     + (f", of which {sum(r[2] for r in rows)} contain an equidistant kept point." if sets else "."))
        for seed, k, eq in rows:
            lines.append(f"    seed {seed}: {k} failing point{'s' if k > 1 else ''}, equidistant kept point "
                         f"{'present' if eq else 'absent'}")
    return lines


def main():
    """Run the Remark 1 experiment, 1-NN consistency of the prototypes on the kept points under ties.

    Remark 1 states that with consistent completion every kept point has a prototype of its own class at least as
    close as any prototype of another class, so that 1-NN on the prototypes cannot misclassify it; with pruning on,
    the only exceptions arise at exact ties. Two experiments with fixed seeds, each dataset fitted with consistent
    completion and pruning off and on: tie-heavy, N_TIE = 400 small datasets with many duplicate rows and tied
    distances (tie_heavy_dataset), and continuous, N_CONT = 1000 datasets with standard normal coordinates and no
    duplicate rows (continuous_dataset); purity, s_min and eps_scale vary per dataset (settings). For each, the
    failing kept points (failures) are counted and every dataset with a failure is checked for an equidistant kept
    point (equidistant). tests/test_remark1.py runs smaller versions of both as part of the test suite.

    Run from this directory: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_remark1.py
    No other script is required first; it runs in a single process.

    Args:
        None (no command-line arguments are read).

    Returns:
        None. Writes results/remark1.txt and prints the text.
    """
    RESULTS.mkdir(exist_ok=True)
    lines = ["Remark 1: 1-NN consistency of the prototypes on the kept points. A failure is a kept point with a "
             "prototype of another class strictly closer than every prototype of its own class; an equidistant "
             "kept point has, at some single distance, points of K u G of two different labels.", ""]
    lines += report("Tie-heavy", N_TIE, run(tie_heavy_dataset, range(N_TIE))) + [""]
    lines += report("Continuous", N_CONT, run(continuous_dataset, range(N_CONT)))
    text = "\n".join(lines) + "\n"
    (RESULTS / "remark1.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

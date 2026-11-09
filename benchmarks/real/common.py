from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# The twelve PMLB files of the earlier evaluation, wdbc from scikit-learn, and five more PMLB datasets
# (ecoli, vowel, waveform_21, wine_quality_red, yeast).
DATASETS = ["bupa", "ecoli", "ionosphere", "new_thyroid", "page_blocks", "phoneme", "ring", "satimage",
            "segmentation", "sonar", "texture", "twonorm", "vehicle", "vowel", "waveform_21", "wdbc",
            "wine_quality_red", "yeast"]
# Largest over smallest class size at least 3.
IMBALANCED = ["ecoli", "new_thyroid", "page_blocks", "wine_quality_red", "yeast"]

# Grids tuned by inner cross-validation, in tie-break order. Methods with a single configuration are not tuned.
K_VALUES = (1, 2, 3, 4, 5, 7)
GRIDS = {
    "full": [("full", {})],
    "pens": [(f"l{lam:g}_{completion}", dict(eps_scale=lam, completion=completion))
             for lam in (1.0, 2.0, 4.0, 8.0, 16.0) for completion in ("gain", "consistent")],
    "enn": [(f"k{k}", {"k": k}) for k in K_VALUES],
    "cnn": [(f"k{k}", {"k": k}) for k in K_VALUES],
    "drop3": [(f"k{k}", {"k": k}) for k in K_VALUES],
    "icf": [(f"k{k}", {"k": k}) for k in K_VALUES],
    "ib3": [(f"a{a:g}_d{d:g}", {"accept": a, "drop": d}) for a in (0.8, 0.9) for d in (0.6, 0.7, 0.8)],
    "hmnei": [(f"eps{e:g}", {"epsilon": e}) for e in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)],
    "lsbo": [("default", {})],
    # eps at quantiles of the pairwise distances, up to the 14th percentile as in Bien and Tibshirani (2011).
    "ps": [(f"q{q:g}", {"q": q}) for q in (0.005, 0.01, 0.02, 0.04, 0.08, 0.14)],
    # gamma at quantiles of the nearest-enemy distances; q = 0 is the margin of the sample.
    "gkn": [(f"q{q:g}", {"q": q}) for q in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)],
}
# Fitted at the per-class sizes of the selected PENS configuration.
MATCHED = ["random", "spot"]
N_RANDOM_DRAWS = 5


def load(name):
    """Load one dataset of the real-data arm as a feature matrix and a label vector.

    Args:
        name: Dataset name from DATASETS; "wdbc" is loaded from scikit-learn, every other name from
            data/<name>.tsv.gz (a PMLB file whose label column is "target").

    Returns:
        Tuple (X, y): X is the (n, d) float feature array, y the (n,) array of class labels.
    """
    if name == "wdbc":
        d = load_breast_cancer()
        return d.data.astype(float), d.target
    df = pd.read_csv(HERE / "data" / f"{name}.tsv.gz", sep="\t")
    return df.drop(columns="target").to_numpy(dtype=float), df["target"].to_numpy()


def outer_folds(X, y):
    """Outer stratified 5-fold split (shuffled, random_state=0) shared by every script of the real-data arm.

    Args:
        X: (n, d) feature array.
        y: (n,) class labels.

    Returns:
        List of five (train_indices, test_indices) pairs of row indices into X.
    """
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))


def inner_folds(X, y):
    """Inner stratified 5-fold split (shuffled, random_state=1) of an outer training part, used for tuning.

    Args:
        X: (m, d) feature array of the outer training part.
        y: (m,) class labels.

    Returns:
        List of five (train_indices, validation_indices) pairs of row indices into X.
    """
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(X, y))


def grid(method, X, y):
    """List the configurations of `method` that are tuned on dataset (X, y).

    A configuration with k neighbors is left out when a class of some inner training part has k points or fewer,
    since no point of that class can then have k neighbors of its own class (on the 18 datasets this drops k = 7 on
    wine_quality_red).

    Args:
        method: Key of GRIDS.
        X: (n, d) feature array of the whole dataset.
        y: (n,) class labels.

    Returns:
        List of (label, kwargs) pairs from GRIDS[method], in grid order, less the excluded k values.
    """
    smallest = min(np.unique(y[tr][itr], return_counts=True)[1].min()
                   for tr, _ in outer_folds(X, y) for itr, _ in inner_folds(X[tr], y[tr]))
    return [(label, cfg) for label, cfg in GRIDS[method] if cfg.get("k", 0) < smallest]


def minmax(Xtr, Xte):
    """Scale both parts by the per-feature range of the training part; constant features map to 0.

    Args:
        Xtr: (m, d) training features.
        Xte: (t, d) test features, scaled with the statistics of Xtr.

    Returns:
        Tuple (Xtr_scaled, Xte_scaled); every feature of the training part lies in [0, 1].
    """
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (Xtr - lo) / span, (Xte - lo) / span


def bal_acc(XS, yS, Xte, yte):
    """Balanced accuracy of 1-NN on the reduced set (XS, yS), evaluated on (Xte, yte).

    Args:
        XS: (k, d) features of the selected rows.
        yS: (k,) their labels.
        Xte: (t, d) test features.
        yte: (t,) test labels.

    Returns:
        Balanced accuracy as a float in [0, 1].
    """
    pred = KNeighborsClassifier(n_neighbors=1).fit(XS, yS).predict(Xte)
    return float(balanced_accuracy_score(yte, pred))


def pick_tradeoff(inner_bal_acc, inner_reduction):
    """Pick one configuration by the trade-off rule on the inner-CV scores.

    Among configurations not dominated on (inner balanced accuracy, inner reduction), the one closest to the ideal
    point after rank-normalizing both; ties go to the first in grid order. Configurations with identical scores count
    once, as the first of them in grid order, so that duplicates do not shift the ranks.

    Args:
        inner_bal_acc: (g,) inner-CV balanced accuracies of the configurations, in grid order.
        inner_reduction: (g,) inner-CV reductions, in the same order.

    Returns:
        Integer position of the picked configuration in the input arrays.
    """
    f = np.column_stack([inner_bal_acc, inner_reduction])
    first = [i for i in range(len(f)) if not any((f[j] == f[i]).all() for j in range(i))]
    front = [i for i in first if not any((f[j] >= f[i]).all() and (f[j] > f[i]).any() for j in first)]
    m = len(front)
    if m == 1:
        return front[0]
    ranks = np.empty((m, 2))
    for j in range(2):
        ranks[np.argsort(f[front, j], kind="stable"), j] = np.arange(m)  # rank 0 = worst
    return front[int(np.argmin(np.linalg.norm(ranks / (m - 1) - 1.0, axis=1)))]

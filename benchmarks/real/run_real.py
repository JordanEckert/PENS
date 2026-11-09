import csv
import sys
import time

import numpy as np

import baselines as B
from common import (DATASETS, GRIDS, N_RANDOM_DRAWS, RESULTS, bal_acc, grid, inner_folds, load, minmax,
                    outer_folds, pick_tradeoff)
from pens import PENS

COLUMNS = ["dataset", "fold", "method", "config", "n_train", "n_selected", "bal_acc", "reduction", "time_s",
           "inner_bal_acc", "inner_reduction"]
NAN = float("nan")


def select(method, cfg, X, y):
    """Fit one selector on a training part.

    Args:
        method: "full" (keeps every row), "pens", or one of the baselines enn, cnn, drop3, icf, ib3, hmnei, lsbo,
            ps, gkn.
        cfg: Keyword arguments of the configuration, passed to PENS or to the baseline function.
        X: (m, d) scaled training features.
        y: (m,) training labels.

    Returns:
        Tuple (sorted row indices into X of the selected rows, fitted PENS model or None for every other method).
    """
    if method == "full":
        return np.arange(len(X)), None
    if method == "pens":
        m = PENS(**cfg).fit(X, y)
        return m.prototype_indices_, m
    fn = {"enn": B.enn, "cnn": B.cnn, "drop3": B.drop3, "icf": B.icf, "ib3": B.ib3, "hmnei": B.hmnei,
          "lsbo": B.lsbo, "ps": B.ps, "gkn": B.gkn}[method]
    return fn(X, y, **cfg), None


def inner_scores(method, cfg, X, y):
    """Score one configuration by stratified 5-fold inner cross-validation on an outer training part.

    Each inner training part is min-max scaled (the validation part with its statistics), the selector is fitted
    on it, and 1-NN on the selection is scored on the validation part.

    Args:
        method: Method key of common.GRIDS.
        cfg: Keyword arguments of the configuration.
        X: (m, d) outer training features.
        y: (m,) their labels.

    Returns:
        Tuple (mean inner balanced accuracy, mean inner reduction) over the five inner folds.
    """
    bal, red = [], []
    for itr, iva in inner_folds(X, y):
        Xa, Xv = minmax(X[itr], X[iva])
        idx, _ = select(method, cfg, Xa, y[itr])
        bal.append(bal_acc(Xa[idx], y[itr][idx], Xv, y[iva]))
        red.append(1.0 - len(idx) / len(itr))
    return float(np.mean(bal)), float(np.mean(red))


def matched(name, fold, Xtr, ytr, Xte, yte, counts):
    """Fit and score the size-matched baselines at the given per-class counts.

    Random selection is drawn N_RANDOM_DRAWS times with seeds (index of the dataset in DATASETS, fold, draw) and its
    balanced accuracy and fit time are averaged over the draws; SPOTgreedy runs once. Both rows share n_train,
    n_selected and reduction.

    Args:
        name: Dataset name.
        fold: Outer fold index.
        Xtr: (m, d) scaled training features.
        ytr: (m,) training labels.
        Xte: (t, d) scaled test features.
        yte: (t,) test labels.
        counts: Dict mapping class label to the number of prototypes PENS selected of that class.

    Returns:
        List of two row dicts (method "random" and "spot") with n_train, n_selected, reduction, bal_acc and time_s.
    """
    n, k = len(Xtr), sum(counts.values())
    base = {"n_train": n, "n_selected": k, "reduction": 1.0 - k / n}
    scores, times = [], []
    for d in range(N_RANDOM_DRAWS):
        rng = np.random.default_rng([DATASETS.index(name), fold, d])
        t0 = time.perf_counter()
        idx = B.random_matched(ytr, counts, rng)
        times.append(time.perf_counter() - t0)
        scores.append(bal_acc(Xtr[idx], ytr[idx], Xte, yte))
    rows = [{"method": "random", **base, "bal_acc": float(np.mean(scores)), "time_s": float(np.mean(times))}]
    t0 = time.perf_counter()
    idx = B.spot_matched(Xtr, ytr, counts)
    t = time.perf_counter() - t0
    rows.append({"method": "spot", **base, "bal_acc": bal_acc(Xtr[idx], ytr[idx], Xte, yte), "time_s": t})
    return rows


def main():
    """Real-data arm, phase 1: every configuration of every method on every outer fold of the named datasets.

    Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_real.py dataset [dataset ...]
    Reads data/<dataset>.tsv.gz (wdbc from scikit-learn). For each dataset and outer fold (stratified 5-fold), the
    training part is min-max scaled and every configuration of every method in common.GRIDS (less the k values that
    common.grid leaves out on the dataset) is fitted on it and scored by 1-NN on the test part. Methods with more
    than one configuration are also scored by stratified 5-fold inner cross-validation on the training part
    (inner_bal_acc, inner_reduction), from which select_real.py picks one configuration by the trade-off rule. The
    size-matched baselines (random selection and SPOTgreedy) are fitted at the per-class sizes of the PENS
    configuration that the trade-off rule picks on that fold, and carry its config label. time_s is the wall-clock
    time of the selector's fit on the outer training part.

    Args:
        sys.argv[1:]: Names of the datasets to run, from common.DATASETS; one shard per dataset.

    Returns:
        None. Writes results/shards/<dataset>.csv (columns COLUMNS), flushed after every row, and prints the time
        of every fold.
    """
    out_dir = RESULTS / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        X, y = load(name)
        grids = {method: grid(method, X, y) for method in GRIDS}
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()

            def emit(row):
                """Write one row to the shard, NaN for the columns it lacks, and flush the file.

                Args:
                    row: Dict with a subset of COLUMNS.

                Returns:
                    None.
                """
                w.writerow({c: row.get(c, NAN) for c in COLUMNS})
                f.flush()

            for fold, (tr, te) in enumerate(outer_folds(X, y)):
                t_fold = time.perf_counter()
                Xtr, Xte = minmax(X[tr], X[te])
                ytr, yte = y[tr], y[te]
                for method, configs in grids.items():
                    pens_rows = []
                    for label, cfg in configs:
                        t0 = time.perf_counter()
                        idx, model = select(method, cfg, Xtr, ytr)
                        t = time.perf_counter() - t0
                        row = {"dataset": name, "fold": fold, "method": method, "config": label, "n_train": len(tr),
                               "n_selected": len(idx), "bal_acc": bal_acc(Xtr[idx], ytr[idx], Xte, yte),
                               "reduction": 1.0 - len(idx) / len(tr), "time_s": t}
                        if len(configs) > 1:
                            row["inner_bal_acc"], row["inner_reduction"] = inner_scores(method, cfg, Xtr, ytr)
                        if method == "pens":
                            pens_rows.append((row, {c: len(model.prototypes_[c]) for c in model.classes_}))
                        emit(row)
                    if method == "pens":
                        pick = pick_tradeoff(np.array([r["inner_bal_acc"] for r, _ in pens_rows]),
                                             np.array([r["inner_reduction"] for r, _ in pens_rows]))
                        best, counts = pens_rows[pick]
                        for row in matched(name, fold, Xtr, ytr, Xte, yte, counts):
                            emit({"dataset": name, "fold": fold, "config": best["config"], **row})
                print(f"{name} fold {fold} done in {time.perf_counter() - t_fold:.0f}s", flush=True)


if __name__ == "__main__":
    main()

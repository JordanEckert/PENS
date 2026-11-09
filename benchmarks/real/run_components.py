import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import run_real
from common import GRIDS, RESULTS, bal_acc, load, minmax, outer_folds
from fps import fps_matched
from pens import PENS

NOEDIT = dict(purity=1.0, s_min=1)
LAMBDAS = (1.0, 2.0, 4.0, 8.0, 16.0)
GRID = {"noedit": [(label, {**cfg, **NOEDIT}) for label, cfg in GRIDS["pens"]],
        "nocomp": [(f"l{lam:g}", dict(eps_scale=lam, completion=None)) for lam in LAMBDAS]}
PENS_CONFIGS = dict(GRIDS["pens"])
NAN = float("nan")
STRICT = os.environ.get("STRICT", "1") != "0"
REFIT_COLUMNS = ["dataset", "fold", "config", "n_selected", "n_refit", "duplicate_rows", "tied_distances"]

_select = run_real.select


def select(method, cfg, X, y):
    """run_real.select, asserting that a fit without the edit keeps every training row.

    With purity=1.0 only pure nodes pass the purity test, and with s_min=1 only single rows pass the size rule, so
    every accepted node is pure and no row is removed. Installed in place of run_real.select below, so that the
    inner fits of run_real.inner_scores are checked as well.

    Args:
        method: Method key as in run_real.select ("pens" for every variant of this script).
        cfg: Keyword arguments of the configuration.
        X: (m, d) scaled training features.
        y: (m,) training labels.

    Returns:
        Tuple (sorted row indices into X, fitted PENS model or None), as run_real.select.

    Raises:
        AssertionError: If a configuration with purity 1.0 and s_min 1 removed a training row.
    """
    idx, model = _select(method, cfg, X, y)
    if model is not None and cfg.get("purity") == 1.0 and cfg.get("s_min") == 1:
        kept = sum(len(model.kept_[c]) for c in model.classes_)
        assert kept == len(X), f"noedit fit removed {len(X) - kept} of {len(X)} rows"
    return idx, model


run_real.select = select  # run_real.inner_scores then checks the inner fits as well


def fit_row(name, fold, method, label, cfg, Xtr, ytr, Xte, yte, tuned):
    """Fit one variant configuration (a PENS fit) on an outer fold and build its result row.

    The row has the columns of run_real.COLUMNS: time_s is the wall-clock time of the fit on the outer training part,
    bal_acc the balanced accuracy of 1-NN on the test part, reduction = 1 - n_selected / n_train; with `tuned`, the
    inner-CV columns are added by run_real.inner_scores.

    Args:
        name: Dataset name.
        fold: Outer fold index.
        method: Variant name written to the row ("noedit" or "nocomp").
        label: Configuration label written to the row (a grid label or "pens_selected").
        cfg: PENS keyword arguments of the variant configuration.
        Xtr: (m, d) scaled training features.
        ytr: (m,) training labels.
        Xte: (t, d) scaled test features.
        yte: (t,) test labels.
        tuned: Whether to add inner_bal_acc and inner_reduction.

    Returns:
        Dict with the row's values (the inner columns only when tuned).
    """
    t0 = time.perf_counter()
    idx, _ = select("pens", cfg, Xtr, ytr)
    t = time.perf_counter() - t0
    row = {"dataset": name, "fold": fold, "method": method, "config": label, "n_train": len(Xtr),
           "n_selected": len(idx), "bal_acc": bal_acc(Xtr[idx], ytr[idx], Xte, yte),
           "reduction": 1.0 - len(idx) / len(Xtr), "time_s": t}
    if tuned:
        row["inner_bal_acc"], row["inner_reduction"] = run_real.inner_scores("pens", cfg, Xtr, ytr)
    return row


def main():
    """Real-data arm, component comparison (A), phase 1: PENS with one stage removed, on every outer fold.

    Run after select_real.py:
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_components.py dataset [...]
    Reads results/real.csv (the PENS configuration and n_selected of every fold) and the datasets. Three variants,
    each reached through existing PENS parameters or the size-matched farthest-point rule, and each tuned or matched
    exactly as PENS is in run_real.py:
    - noedit: Stage 1 (the dendrogram edit) removed by purity=1.0 and s_min=1 (NOEDIT), so that no row is removed
      (checked by select). The radius eps = eps_scale x median merge height of K changes with it, since K = X here.
      Grid: the ten PENS configurations.
    - nocomp: Stage 3 (completion) removed by completion=None. Grid: eps_scale in {1, 2, 4, 8, 16}.
    - fps: edit and completion removed. Class-wise farthest-point selection (fps.fps_matched) at the per-class sizes
      of the PENS configuration that the trade-off rule selected on the fold, like the matched baselines of
      run_real.py. Not tuned; time_s excludes the PENS refit, as for SPOTgreedy.
    noedit and nocomp are also fitted at the configuration PENS selected on the fold (nocomp at its eps_scale, noedit
    at its eps_scale and completion), reported with config "pens_selected", so that the stages can be compared at a
    fixed radius as well as after tuning. PENS itself is not rerun: select_components.py copies its rows from
    results/real.csv. The PENS refit at the selected configuration is checked against n_selected of real.csv. On
    datasets with many exactly tied distances (duplicate rows or quantised features) the gain completion resolves
    ties by floating-point comparison, so a refit on another machine can differ by a few prototypes. With STRICT=1
    (the default) such a mismatch stops the run; with STRICT=0 it is recorded, and fps is matched to the refit. Fit
    times are comparable with results/summary.txt only on the machine of the reported run_real.py runs.

    Args:
        sys.argv[1:]: Names of the datasets to run, from common.DATASETS; one shard per dataset.

    Returns:
        None. Writes results/components_shards/<dataset>.csv (the columns of run_real.COLUMNS) and
        results/components_shards/<dataset>_refit.csv (REFIT_COLUMNS: one row per fold with the stored and the
        refitted prototype count, the number of duplicate training rows and the number of repeated pairwise
        distances, which select_components.py reports), both flushed after every row; prints the time of every fold.

    Raises:
        AssertionError: With STRICT=1, if the refit does not reproduce n_selected of real.csv; in any case, if the
            farthest-point selection does not have the per-class counts of the refit.
    """
    real = pd.read_csv(RESULTS / "real.csv")
    pens = real[real.method == "pens"].set_index(["dataset", "fold"])
    out_dir = RESULTS / "components_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        X, y = load(name)
        with open(out_dir / f"{name}.csv", "w", newline="") as f, \
                open(out_dir / f"{name}_refit.csv", "w", newline="") as g:
            w = csv.DictWriter(f, fieldnames=run_real.COLUMNS)
            w.writeheader()
            refit = csv.DictWriter(g, fieldnames=REFIT_COLUMNS)
            refit.writeheader()

            def emit(row):
                """Write one row to the shard, NaN for the columns it lacks, and flush the file.

                Args:
                    row: Dict with a subset of run_real.COLUMNS.

                Returns:
                    None.
                """
                w.writerow({c: row.get(c, NAN) for c in run_real.COLUMNS})
                f.flush()

            for fold, (tr, te) in enumerate(outer_folds(X, y)):
                t_fold = time.perf_counter()
                Xtr, Xte = minmax(X[tr], X[te])
                ytr, yte = y[tr], y[te]
                selected = pens.loc[(name, fold), "config"]
                n_selected = int(pens.loc[(name, fold), "n_selected"])
                model = PENS(**PENS_CONFIGS[selected]).fit(Xtr, ytr)
                n_refit = len(model.prototype_indices_)
                assert n_refit == n_selected or not STRICT, (name, fold, selected, n_selected, n_refit)
                if n_refit != n_selected:
                    print(f"{name} fold {fold}: the refit at {selected} gives {n_refit} prototypes, real.csv stores "
                          f"{n_selected}", flush=True)
                D = pdist(Xtr)
                refit.writerow(dict(dataset=name, fold=fold, config=selected, n_selected=n_selected, n_refit=n_refit,
                                    duplicate_rows=len(Xtr) - len(np.unique(Xtr, axis=0)),
                                    tied_distances=len(D) - len(np.unique(D))))
                g.flush()
                del D
                counts = {c: len(model.prototypes_[c]) for c in model.classes_}
                for method, configs in GRID.items():
                    for label, cfg in configs:
                        emit(fit_row(name, fold, method, label, cfg, Xtr, ytr, Xte, yte, tuned=True))
                same = {"noedit": {**PENS_CONFIGS[selected], **NOEDIT},
                        "nocomp": dict(eps_scale=PENS_CONFIGS[selected]["eps_scale"], completion=None)}
                for method, cfg in same.items():
                    emit(fit_row(name, fold, method, "pens_selected", cfg, Xtr, ytr, Xte, yte, tuned=False))
                t0 = time.perf_counter()
                idx = fps_matched(Xtr, ytr, counts)
                t = time.perf_counter() - t0
                assert len(idx) == n_refit and all((ytr[idx] == c).sum() == k for c, k in counts.items())
                emit({"dataset": name, "fold": fold, "method": "fps", "config": selected, "n_train": len(tr),
                      "n_selected": len(idx), "bal_acc": bal_acc(Xtr[idx], ytr[idx], Xte, yte),
                      "reduction": 1.0 - len(idx) / len(tr), "time_s": t})
                print(f"{name} fold {fold} done in {time.perf_counter() - t_fold:.0f}s", flush=True)


if __name__ == "__main__":
    main()

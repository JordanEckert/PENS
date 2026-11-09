import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import DATASETS, GRIDS, RESULTS, bal_acc, load, minmax, outer_folds
from pens import PENS

CONFIGS = dict(GRIDS["pens"])
# Setting: (default, other values).
VARIANTS = {
    "purity": (0.9, [0.8, 0.95, 1.0, 0.75, 0.5]),
    "s_min": (5, [1, 2, 10]),
    "min_keep": (1, [2, 3]),
    "gain": (1.0, [0.5, 2.0]),
    "class_weight": ("balanced", [None]),
    "prune": (True, [False]),
}
assert all(getattr(PENS(), p) == d for p, (d, _) in VARIANTS.items()), "defaults differ from PENS()"


def task(args):
    """Sensitivity rows of one (dataset, outer fold).

    PENS is refitted on the min-max scaled outer training part at the configuration that select_real.py picked,
    first at the defaults (checked against n_selected of real.csv) and then with one fixed setting of VARIANTS
    changed at a time to each of its other values, and scored by 1-NN on the outer test part. gain acts only with
    gain completion, so its rows exist only on folds where gain completion was picked; prune acts only with
    completion on, which every grid configuration has.

    Args:
        args: Tuple (dataset name, fold index, label of the selected configuration, its n_selected in real.csv).

    Returns:
        List of row dicts with dataset, fold, config, param ("default" or the setting), value (as a string),
        n_selected, bal_acc, reduction, effectiveness, kept_share (|K| / n_train) and guarded (number of classes
        whose kept set is empty).

    Raises:
        AssertionError: If the refit at the defaults does not give n_selected prototypes.
    """
    name, fold, config, n_selected = args
    X, y = load(name)
    tr, te = outer_folds(X, y)[fold]
    Xtr, Xte = minmax(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    base = CONFIGS[config]
    runs = [("default", "default", base)]
    for p, (_, values) in VARIANTS.items():
        if p == "gain" and base["completion"] != "gain":
            continue
        runs += [(p, str(v), {**base, p: v}) for v in values]
    rows = []
    for param, value, cfg in runs:
        m = PENS(**cfg).fit(Xtr, ytr)
        idx = m.prototype_indices_
        if param == "default":
            assert len(idx) == n_selected, (name, fold, config)
        reduction = 1.0 - len(idx) / len(tr)
        acc = bal_acc(Xtr[idx], ytr[idx], Xte, yte)
        rows.append(dict(dataset=name, fold=fold, config=config, param=param, value=value, n_selected=len(idx),
                         bal_acc=acc, reduction=reduction, effectiveness=acc * reduction,
                         kept_share=sum(len(k) for k in m.kept_.values()) / len(tr),
                         guarded=sum(len(m.kept_[c]) == 0 for c in m.classes_)))
    return rows


def summary(df):
    """Format the text of results/sensitivity_real.txt.

    One table per setting of VARIANTS: for the default and each other value, the number of folds, means over
    datasets of the fold means of balanced accuracy, reduction, effectiveness and kept share (%), the differences
    to the defaults in balanced accuracy and effectiveness (points, on the same folds), the datasets where balanced
    accuracy is higher, equal or lower than at the defaults, and the folds with a class whose kept set is empty.

    Args:
        df: Rows as written to results/sensitivity_real.csv.

    Returns:
        The summary text (no trailing newline).
    """
    lines = ["Sensitivity of PENS to its fixed settings, one at a time, at the configuration selected on each fold.",
             "Means over datasets of fold means (%); differences to the defaults in points on the same folds; "
             "datasets where balanced accuracy is higher, equal or lower than at the defaults; folds with a class "
             "whose kept set is empty.", ""]
    default = df[df.param == "default"].set_index(["dataset", "fold"])
    metrics = ["bal_acc", "reduction", "effectiveness", "kept_share"]
    for p, (d, values) in VARIANTS.items():
        sub = df[df.param == p]
        if sub.empty:
            continue
        keys = sorted(set(zip(sub.dataset, sub.fold)))
        base = default.loc[keys].reset_index()
        bmean = base.groupby("dataset")[metrics].mean()
        out = [dict(value=f"{d} (default)", folds=len(keys), **(100 * bmean.mean()).to_dict(), d_bal=0.0, d_eff=0.0,
                    wtl="", guarded=int((base.guarded > 0).sum()))]
        for v in values:
            g = sub[sub.value == str(v)]
            gmean = g.groupby("dataset")[metrics].mean()
            diff = (gmean.bal_acc - bmean.bal_acc).loc[gmean.index]
            out.append(dict(value=str(v), folds=len(g), **(100 * gmean.mean()).to_dict(),
                            d_bal=100 * diff.mean(), d_eff=100 * (gmean.effectiveness - bmean.effectiveness).mean(),
                            wtl=f"{(diff > 1e-12).sum()}-{(diff.abs() <= 1e-12).sum()}-{(diff < -1e-12).sum()}",
                            guarded=int((g.guarded > 0).sum())))
        t = pd.DataFrame(out).set_index("value")
        t.columns = ["folds", "bal. acc.", "reduction", "effect.", "kept share", "d bal. acc.", "d effect.",
                     "datasets +/=/-", "guarded folds"]
        lines += [f"{p}" + (" (folds with gain completion)" if p == "gain" else ""),
                  t.to_string(float_format=lambda v: f"{v:.1f}"), ""]
    return "\n".join(lines)


def main():
    """Real-data arm, phase 5: sensitivity of PENS to the settings that are not tuned.

    Run after select_real.py: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python sensitivity_real.py (no arguments).
    Reads results/real.csv (the selected PENS configuration and n_selected of every fold) and the datasets, runs
    task on every (dataset, fold) in a process pool and sorts the rows by dataset and fold. Nothing is selected from
    these runs; they show how much the results depend on each setting. NPROC (default 4) sets the number of
    processes.

    Returns:
        None. Writes results/sensitivity_real.csv (one row per dataset, fold, setting and value) and
        results/sensitivity_real.txt, which is also printed.
    """
    real = pd.read_csv(RESULTS / "real.csv")
    pens = real[real.method == "pens"].set_index(["dataset", "fold"])
    tasks = [(n, f, pens.loc[(n, f), "config"], int(pens.loc[(n, f), "n_selected"])) for n in DATASETS for f in range(5)]
    rows = []
    with Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool:
        for out in pool.imap_unordered(task, tasks):
            rows.extend(out)
            print(f"{out[0]['dataset']} fold {out[0]['fold']} done", flush=True)
    df = pd.DataFrame(rows)
    df = df.iloc[np.lexsort((df.fold.to_numpy(), df.dataset.map(DATASETS.index).to_numpy()))]
    df.to_csv(RESULTS / "sensitivity_real.csv", index=False)
    text = summary(df)
    (RESULTS / "sensitivity_real.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

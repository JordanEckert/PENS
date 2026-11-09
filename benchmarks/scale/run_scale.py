import os
import resource
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "real"))

import baselines as B  # noqa: E402
from common import bal_acc, minmax  # noqa: E402
from pens import PENS  # noqa: E402

DATASETS = ["magic", "letter"]
SIZES = [1000, 2000, 5000, 10000, 15000]
TEST = 4000
SEED = 0
METHODS = ["pens", "full", "enn", "cnn", "drop3", "icf", "ib3", "hmnei", "lsbo", "ps", "gkn", "random", "spot"]
MATCHED = {"random", "spot"}


def split(name, n):
    """Min-max scaled training subset of n pool rows and the test set, both stratified; fixed for (name, n).

    The dataset is split once into a stratified test set of TEST rows and a pool (random_state=SEED); when n is
    smaller than the pool, a stratified subset of n rows is drawn from it (random_state=SEED). Features are min-max
    scaled by the range of the training subset (common.minmax of the real-data arm).

    Args:
        name: Dataset name; read from data/{name}.tsv.gz, a PMLB file whose label column is "target".
        n: Requested training size.

    Returns:
        Tuple (Xtr, ytr, Xte, yte) of the scaled training features, training labels, scaled test features and test
        labels.
    """
    df = pd.read_csv(HERE / "data" / f"{name}.tsv.gz", sep="\t")
    X, y = df.drop(columns="target").to_numpy(dtype=float), df["target"].to_numpy()
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=TEST, stratify=y, random_state=SEED)
    if n < len(yp):
        Xp, _, yp, _ = train_test_split(Xp, yp, train_size=n, stratify=yp, random_state=SEED)
    Xtr, Xte = minmax(Xp, Xte)
    return Xtr, yp, Xte, yte


def select(method, X, y, counts):
    """Run one selector at its fixed settings and return the selected row indices.

    There is no inner cross-validation: PENS uses eps_scale = 4 (the package default) with gain completion (the
    completion of the real-data grid); "full" keeps every row; random selection and SPOTgreedy use the per-class
    sizes of PENS (baselines.random_matched with default_rng(SEED), baselines.spot_matched); the other competitors
    use their usual defaults, PS at the 2% quantile of the pairwise distances and GKN at the 10% quantile of the
    nearest-enemy distances (the middle of their real-data grids).

    Args:
        method: One of METHODS.
        X: (n, d) scaled training features.
        y: (n,) training labels.
        counts: Dict class -> number of PENS prototypes, for the methods in MATCHED; None otherwise.

    Returns:
        Array of selected row indices into X.
    """
    if method == "pens":
        return PENS(eps_scale=4.0, completion="gain").fit(X, y).prototype_indices_
    if method == "full":
        return np.arange(len(X))
    if method == "random":
        return B.random_matched(y, counts, np.random.default_rng(SEED))
    if method == "spot":
        return B.spot_matched(X, y, counts)
    return {"enn": B.enn, "cnn": B.cnn, "drop3": B.drop3, "icf": B.icf, "ib3": B.ib3, "hmnei": B.hmnei,
            "lsbo": B.lsbo, "ps": lambda X, y: B.ps(X, y, q=0.02), "gkn": lambda X, y: B.gkn(X, y, q=0.1)}[method](X, y)


def peak_mb():
    """Peak resident memory of the current process so far, in MB.

    Args:
        None (no arguments).

    Returns:
        ru_maxrss of the process converted to MB; it is reported in bytes on macOS and in kilobytes on Linux.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10  # bytes on macOS, kilobytes on Linux


def fit(args):
    """One fit, run in its own process so that its time is not shared and its peak memory can be read.

    Args:
        args: Tuple (dataset name, n, method, counts), with counts the per-class sizes of PENS for the methods in
            MATCHED and None otherwise.

    Returns:
        Dict with dataset, n (the rows actually used), method, n_selected, reduction = 1 - n_selected / n, bal_acc
        (balanced accuracy of 1-NN on the selected rows, measured on the test set), time_s (the fit alone), peak_mb
        (peak resident memory of the process after the fit), extra_mb (that peak minus the peak before the fit, so it
        approximates the memory the selector itself needs) and counts, the per-class sizes of the selection.
    """
    name, n, method, counts = args
    Xtr, ytr, Xte, yte = split(name, n)
    before = peak_mb()
    t0 = time.perf_counter()
    idx = np.asarray(select(method, Xtr, ytr, counts))
    t = time.perf_counter() - t0
    after = peak_mb()
    labels, sizes = np.unique(ytr[idx], return_counts=True)
    return dict(dataset=name, n=len(ytr), method=method, n_selected=len(idx), reduction=1.0 - len(idx) / len(ytr),
                bal_acc=bal_acc(Xtr[idx], ytr[idx], Xte, yte), time_s=t, peak_mb=after, extra_mb=after - before,
                counts=dict(zip(labels.tolist(), sizes.tolist())))


def summary(df):
    """Compose results/summary.txt: per dataset, tables of fit time, extra memory, reduction and balanced accuracy.

    Each table has one row per method (METHODS order) and one column per training size, reduction and balanced
    accuracy in percent; after them, the exponent b of fit time ~ n^b, fitted by least squares in log-log over the
    sizes n >= 2000 for every method but full and random.

    Args:
        df: DataFrame of the fit rows without the counts column (the content of results/scale.csv).

    Returns:
        The summary text.
    """
    lines = [f"Scalability arm: one stratified training subset per size, test set of {TEST} rows, fixed settings.", ""]
    for name in DATASETS:
        d = df[df.dataset == name]
        for col, title, fmt in [("time_s", "fit time (s)", "{:.2f}"), ("extra_mb", "extra peak memory (MB)", "{:.0f}"),
                                ("reduction", "reduction (%)", "{:.1f}"), ("bal_acc", "balanced accuracy (%)", "{:.1f}")]:
            t = d.pivot(index="method", columns="n", values=col).reindex(METHODS)
            if col in ("reduction", "bal_acc"):
                t = 100 * t
            lines += [f"{name}: {title}", t.to_string(float_format=lambda v: fmt.format(v)), ""]
        big = d[d.n >= 2000]
        slopes = {m: np.polyfit(np.log(g.n), np.log(g.time_s.clip(lower=1e-4)), 1)[0]
                  for m, g in big.groupby("method") if m not in ("full", "random") and len(g) >= 2}
        lines += [f"{name}: exponent b of fit time ~ n^b, least squares over n >= 2000",
                  "  " + ", ".join(f"{m} {slopes[m]:.2f}" for m in METHODS if m in slopes), ""]
    return "\n".join(lines)


def main():
    """Run the scalability arm: fit time, peak memory, reduction and balanced accuracy as the training set grows.

    Two PMLB datasets larger than any in the real-data arm, magic (19020 rows, 10 features, 2 classes) and letter
    (20000 rows, 16 features, 26 classes), are each split once into a stratified test set of TEST = 4000 rows and a
    pool. For every n in SIZES (1000 to 15000) a stratified subset of n pool rows is drawn and min-max scaled by its
    own range (split), and every selector of METHODS is fitted on it at fixed settings (select), each fit in a fresh
    spawned process, one at a time, so that times are not shared and its peak memory can be read (fit). PENS is
    fitted first at each size so that its per-class counts can be handed to random selection and SPOTgreedy.
    results/scale.csv is rewritten after every fit, so a partial run leaves a usable file.

    Run from this directory:
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_scale.py
    MAX_N caps the training sizes (for machines with less than about 8 GB of free memory). Reads data/magic.tsv.gz
    and data/letter.tsv.gz and imports baselines and common from ../real. Fit times depend on the machine.

    Args:
        None (no command-line arguments are read); MAX_N is read from the environment.

    Returns:
        None. Writes results/scale.csv (columns dataset, n, method, n_selected, reduction, bal_acc, time_s, peak_mb,
        extra_mb) and results/summary.txt (the text of summary); prints progress and the text.
    """
    RESULTS.mkdir(exist_ok=True)
    max_n = int(os.environ.get("MAX_N", max(SIZES)))
    rows = []
    # One process per fit, one fit at a time, so that times are not shared and peak memory is per fit.
    ctx = get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        for name in DATASETS:
            for n in [s for s in SIZES if s <= max_n]:
                pens = pool.apply(fit, ((name, n, "pens", None),))
                rows.append(pens)
                for method in METHODS[1:]:
                    rows.append(pool.apply(fit, ((name, n, method, pens["counts"] if method in MATCHED else None),)))
                    r = rows[-1]
                    print(f"{name} n={n} {method}: {r['time_s']:.1f}s, {r['extra_mb']:.0f} MB extra", flush=True)
                pd.DataFrame(rows).drop(columns="counts").to_csv(RESULTS / "scale.csv", index=False)
    df = pd.DataFrame(rows).drop(columns="counts")
    df.to_csv(RESULTS / "scale.csv", index=False)
    text = summary(df)
    (RESULTS / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

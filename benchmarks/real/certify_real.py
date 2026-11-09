import os
import sys
from multiprocessing import Pool

import gudhi
import numpy as np
import pandas as pd
import ripser
from scipy.spatial.distance import cdist, pdist, squareform

from common import DATASETS, GRIDS, RESULTS, load, minmax, outer_folds
from pens import PENS, edit_cost, hausdorff_distance

TOL = 1e-9
CONFIGS = dict(GRIDS["pens"])


def diagrams(X):
    """Finite bars of the Vietoris-Rips persistence diagrams of X in degrees 0 and 1 (ripser), and its diameter.

    Args:
        X: (m, d) array of points.

    Returns:
        Tuple ([dgm0, dgm1], diameter): each dgm is a (b, 2) array of finite (birth, death) pairs; for fewer than
        two points both are empty and the diameter is 0.0.
    """
    if len(X) < 2:
        return [np.empty((0, 2)), np.empty((0, 2))], 0.0
    D = squareform(pdist(X))
    dgms = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"]
    return [d[np.isfinite(d[:, 1])] for d in dgms], float(D.max())


def margin(X, y):
    """Smallest distance between differently labeled rows of X (the margin of the sample).

    Args:
        X: (m, d) array of points.
        y: (m,) class labels.

    Returns:
        The margin as a float; inf when y has a single class.
    """
    labels = np.unique(y)
    return min((float(cdist(X[y == a], X[y == b]).min()) for i, a in enumerate(labels) for b in labels[i + 1:]),
               default=np.inf)


def lifetimes(dgm):
    """Bar lengths (death minus birth) of a persistence diagram.

    Args:
        dgm: (b, 2) array of (birth, death) pairs with finite deaths.

    Returns:
        (b,) array of lifetimes.
    """
    return dgm[:, 1] - dgm[:, 0]


def task(args):
    """Certificate rows of one (dataset, outer fold) at every PENS grid configuration.

    PENS is refitted on the min-max scaled outer training part at every configuration of its grid; the kept sets K_c
    do not depend on eps_scale, so their diagrams are computed once. For each configuration and class c the row
    holds the radius eps_c (eps) and 4 eps_c (four_eps), |X_c| (n_c), |K_c| (n_kept), the number of prototypes
    (n_proto), the edit cost eta_c = d_H(K_c, X_c) (eta), the bounds 2 eps_c (bound) and 2 (eps_c + eta_c)
    (e2e_bound), the diameter of K_c (diam_K), d_H(P_c, K_c) (dH_PK) and, in degrees k = 0 and 1 (Vietoris-Rips
    persistence with ripser, finite bars, bottleneck distances with gudhi), d_B(P_c, K_c) (dBk_PK), d_B(P_c, X_c)
    (dBk_PX), the number of finite bars of Dgm(K_c) longer than 4 eps_c (cert_hk, the bars that Corollary 2
    guarantees in Dgm(P_c)) and the longest finite bar of Dgm(K_c) (longest_hk_K). `holds` is Corollary 1,
    d_H(P_c, K_c) <= eps_c and d_B(P_c, K_c) <= 2 eps_c in both degrees, and `e2e_holds` is Theorem 3,
    d_B(P_c, X_c) <= 2 (eps_c + eta_c), each up to TOL. Every row also carries the margins (smallest distance between
    differently labeled points) of the training part and of the kept set, gamma_X and gamma_K (Proposition 5), and
    `selected`, True for the configuration that select_real.py picked on the fold (the one behind the accuracy
    numbers). A class whose kept set is empty gets a row without bounds.

    Args:
        args: Tuple (dataset name, fold index, label of the selected configuration, its n_selected in real.csv).

    Returns:
        List of row dicts, one per (configuration, class).

    Raises:
        AssertionError: If the refit at the selected configuration does not give n_selected prototypes, or the kept
            sets differ between configurations.
    """
    name, fold, selected, n_selected = args
    X, y = load(name)
    tr, te = outer_folds(X, y)[fold]
    Xtr, ytr = minmax(X[tr], X[te])[0], y[tr]
    models = {label: PENS(**cfg).fit(Xtr, ytr) for label, cfg in CONFIGS.items()}
    assert len(models[selected].prototype_indices_) == n_selected, (name, fold, selected)
    first = next(iter(models.values()))
    cost = edit_cost(Xtr, ytr, first.kept_)
    K_all = np.concatenate([np.asarray(first.kept_[c], dtype=int) for c in first.classes_])
    gammas = dict(gamma_X=margin(Xtr, ytr), gamma_K=margin(Xtr[K_all], ytr[K_all]))
    rows = []
    for c in first.classes_:
        X_c, K = np.flatnonzero(ytr == c), np.asarray(first.kept_[c], dtype=int)
        assert all(np.array_equal(m.kept_[c], first.kept_[c]) for m in models.values())
        if len(K):
            dK, diam = diagrams(Xtr[K])
            dX = dK if len(K) == len(X_c) else diagrams(Xtr[X_c])[0]
        for label, m in models.items():
            P, eps = np.asarray(m.prototypes_[c], dtype=int), float(m.radii_[c])
            row = dict(dataset=name, fold=fold, config=label, eps_scale=CONFIGS[label]["eps_scale"],
                       selected=label == selected, cls=c, n_c=len(X_c), n_kept=len(K), n_proto=len(P), eps=eps,
                       four_eps=4 * eps, **gammas)
            if len(K) == 0:
                rows.append(row)
                continue
            dP = diagrams(Xtr[P])[0]
            row.update(eta=cost[c], bound=2 * eps, e2e_bound=2 * (eps + cost[c]), diam_K=diam,
                       dH_PK=hausdorff_distance(Xtr[K], Xtr[P]))
            for k in (0, 1):
                row[f"dB{k}_PK"] = float(gudhi.bottleneck_distance(dP[k], dK[k]))
                row[f"dB{k}_PX"] = float(gudhi.bottleneck_distance(dP[k], dX[k]))
                life = lifetimes(dK[k])
                row[f"cert_h{k}"] = int(np.sum(life > 4 * eps))
                row[f"longest_h{k}_K"] = float(life.max()) if len(life) else 0.0
            row["holds"] = bool(row["dH_PK"] <= eps + TOL and max(row["dB0_PK"], row["dB1_PK"]) <= 2 * eps + TOL)
            row["e2e_holds"] = bool(max(row["dB0_PX"], row["dB1_PX"]) <= row["e2e_bound"] + TOL)
            rows.append(row)
    return rows


def per_fold(rows):
    """Aggregate class rows per (dataset, fold, configuration).

    Args:
        rows: Class rows from task with a nonempty kept set.

    Returns:
        DataFrame with one row per (dataset, fold, config): eps_scale, the largest 4 eps (four_eps), the median
        diameter of K_c (diam), kept and n summed over classes, the largest eta and e2e bound, cert_h0 and cert_h1
        summed over classes, the longest degree-1 bar (longest_h1), rel = four_eps / diam and kept_share = kept / n.
    """
    agg = rows.groupby(["dataset", "fold", "config"], sort=False).agg(
        eps_scale=("eps_scale", "first"), four_eps=("four_eps", "max"), diam=("diam_K", "median"),
        kept=("n_kept", "sum"), n=("n_c", "sum"), eta=("eta", "max"), e2e=("e2e_bound", "max"),
        cert_h0=("cert_h0", "sum"), cert_h1=("cert_h1", "sum"), longest_h1=("longest_h1_K", "max"))
    agg["rel"] = agg.four_eps / agg.diam
    agg["kept_share"] = agg.kept / agg.n
    return agg.reset_index()


def summary(df):
    """Format the text of results/certificate_real.txt.

    It reports how many class rows satisfy Corollary 1 and Theorem 3 (over all configurations and at the selected
    ones) and how many selected class rows have an empty kept set; a per-dataset table at the selected configuration
    (medians over folds) of the most frequent configuration, 4 eps over the median diameter of K_c, kept share,
    largest eta_c, largest end-to-end bound, certified bars in degrees 0 and 1 and the longest degree-1 bar; the
    number of selected folds with a certified degree-1 bar; a profile of all grid configurations by eps_scale; and
    the margins: folds with zero training margin, the median factor gamma_K / gamma_X where it is positive, and per
    eps_scale the folds where eps is below gamma_K with no empty kept set (Proposition 5: completion adds nothing).

    Args:
        df: Class rows as written to results/certificate_real.csv.

    Returns:
        The summary text, ending with a newline.
    """
    ok = df[df.n_kept > 0]
    sel = ok[ok.selected]
    lines = [f"Corollary 1 (degrees 0 and 1) holds in {int(ok.holds.sum())} of {len(ok)} class rows over all "
             f"configurations, and in {int(sel.holds.sum())} of {len(sel)} at the selected ones; Theorem 3 holds in "
             f"{int(ok.e2e_holds.sum())} of {len(ok)} and {int(sel.e2e_holds.sum())} of {len(sel)}. "
             f"{int((df[df.selected].n_kept == 0).sum())} selected class rows have an empty kept set and no bound.", "",
             "At the selected configuration, per dataset (medians over folds): most frequent configuration, 4 eps over "
             "the median diameter of K_c, kept share of the training part, largest edit cost eta_c, largest "
             "end-to-end bound 2 (eps + eta_c), certified bars of all classes in degrees 0 and 1 (finite bars of "
             "Dgm(K_c) longer than 4 eps) and the longest degree-1 bar of any K_c.", ""]
    agg = per_fold(sel)
    t = agg.groupby("dataset", sort=False).agg(
        config=("config", lambda v: v.value_counts().index[0]), rel=("rel", "median"),
        kept_share=("kept_share", "median"), eta=("eta", "median"), e2e=("e2e", "median"),
        cert_h0=("cert_h0", "median"), cert_h1=("cert_h1", "median"), longest_h1=("longest_h1", "median"))
    t.columns = ["config", "4eps/diam", "kept share", "max eta", "max e2e bound", "cert. H0", "cert. H1", "longest H1"]
    lines += [t.to_string(float_format=lambda v: f"{v:.3f}"), "",
              f"Selected configurations with at least one certified degree-1 bar: "
              f"{int((agg.cert_h1 > 0).sum())} of {len(agg)} folds.", "",
              "All grid configurations, by eps_scale (over datasets and folds): median 4 eps over the median "
              "diameter of K_c, share of folds with at least one certified bar in degree 0 and in degree 1, and the "
              "median number of certified degree-0 bars.", ""]
    allf = per_fold(ok)
    prof = allf.groupby("eps_scale").agg(rel=("rel", "median"), h0=("cert_h0", lambda v: (v > 0).mean()),
                                         h1=("cert_h1", lambda v: (v > 0).mean()), n_h0=("cert_h0", "median"))
    prof.columns = ["4eps/diam", "share cert. H0", "share cert. H1", "median cert. H0"]
    lines.append(prof.to_string(float_format=lambda v: f"{v:.3f}"))
    folds = df.groupby(["dataset", "fold", "eps_scale"], sort=False).agg(
        eps=("eps", "max"), empty=("n_kept", lambda v: (v == 0).any()),
        gamma_X=("gamma_X", "first"), gamma_K=("gamma_K", "first"))
    raw = folds.groupby(["dataset", "fold"]).first()
    zero = raw[raw.gamma_X == 0].index.get_level_values("dataset")
    below = folds.assign(b=(folds.eps < folds.gamma_K) & ~folds.empty).groupby("eps_scale").b.sum()
    lines += ["", f"Margin of the training part: zero in {len(zero)} of {len(raw)} folds "
              f"({', '.join(sorted(set(zero))) or 'none'}); over the folds where it is positive, the edit raises it "
              f"by a median factor of {(raw.gamma_K / raw.gamma_X)[raw.gamma_X > 0].median():.2f} "
              "(margin of the kept set over that of the training part).",
              "Folds where eps is below the margin of the kept set and no kept set is empty (Proposition 5: "
              "completion adds nothing), by eps_scale: "
              + ", ".join(f"{k:g}: {int(v)}" for k, v in below.items())]
    return "\n".join(lines) + "\n"


def main():
    """Real-data arm, phase 4: the certificate of PENS on real data, at every grid configuration.

    Run after select_real.py: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python certify_real.py
    Reads results/real.csv (the selected PENS configuration and n_selected of every fold) and the datasets, runs
    task on every (dataset, fold) in a process pool and sorts the rows by dataset, fold, configuration and class.
    NPROC (default 4) sets the number of processes.

    Args:
        sys.argv[1]: Optional; "summary" rewrites results/certificate_real.txt from the committed
            results/certificate_real.csv without refitting. Without arguments every fold is refitted.

    Returns:
        None. Writes results/certificate_real.csv (one row per dataset, fold, configuration and class) and
        results/certificate_real.txt, which is also printed.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        df = pd.read_csv(RESULTS / "certificate_real.csv", keep_default_na=False, na_values=["nan", ""])
        text = summary(df)
        (RESULTS / "certificate_real.txt").write_text(text)
        print(text)
        return
    real = pd.read_csv(RESULTS / "real.csv")
    pens = real[real.method == "pens"].set_index(["dataset", "fold"])
    tasks = [(n, f, pens.loc[(n, f), "config"], int(pens.loc[(n, f), "n_selected"])) for n in DATASETS for f in range(5)]
    rows = []
    with Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool:
        for out in pool.imap_unordered(task, tasks):
            rows.extend(out)
            print(f"{out[0]['dataset']} fold {out[0]['fold']} done", flush=True)
    order = {label: i for i, label in enumerate(CONFIGS)}
    df = pd.DataFrame(rows)
    df = df.iloc[np.lexsort((df.cls.to_numpy(), df.config.map(order).to_numpy(), df.fold.to_numpy(),
                             df.dataset.map(DATASETS.index).to_numpy()))]
    df.to_csv(RESULTS / "certificate_real.csv", index=False)
    text = summary(df)
    (RESULTS / "certificate_real.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

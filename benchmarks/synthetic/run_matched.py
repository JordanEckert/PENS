import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from generators import GENERATORS
from pens import PENS
from topology import HERE, RESULTS, DiagramCache, bott, fps, maxdim, n_longer, top_lifetimes

sys.path.insert(0, str(HERE.parent / "real"))
from stats import ALPHA, holm  # noqa: E402
from vendor.spotgreedy import SPOT_GreedySubsetSelection  # noqa: E402

GENS = list(GENERATORS)
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = (1.0, 4.0, 16.0)
CONFIGS = [(lam, completion) for completion in ("consistent", "gain") for lam in LAMBDAS]
R_RANDOM = 5
METHODS = ["pens", "fps", "spot", "random"]
NAMES = {"pens": "PENS", "fps": "Farthest-point", "spot": "SPOTgreedy", "random": "Random"}
DEG = (0, 1, 2)
# Noise thresholds for long bars: column suffix and description.
THRESHOLDS = {"eps": "", "half": "_half", "matched": "_matched"}
THRESHOLD_NAMES = {"eps": "4 eps_c (certificate)", "half": "half-sample band", "matched": "matched-size band"}
N_SUB, BAND_ALPHA = 100, 0.05
COLS = (["gen", "seed", "eps_scale", "completion", "cls", "method", "beta1_true", "beta2_true", "maxdim", "n_c",
         "n_kept", "n_sel", "eps", "bound", "four_eps", "thr_half", "thr_matched"] + [f"dB{k}_SX" for k in DEG]
        + [f"dB{k}_PK" for k in DEG]
        + [f"nlong{k}_{part}{sfx}" for sfx in THRESHOLDS.values() for part in ("X", "S", "match", "K") for k in DEG]
        + [f"longest{k}_{part}" for part in ("X", "S", "K") for k in DEG]
        + [f"keep{k}{sfx}" for sfx in THRESHOLDS.values() for k in DEG] + ["test_bal_acc"])
KEY = ["gen", "seed", "eps_scale", "completion", "cls"]
NAN = float("nan")
TOL = 1e-9


def band_threshold(D, b, rng, n_sub=N_SUB, alpha=BAND_ALPHA):
    """Noise threshold for long bars by subsampling: 4 times a quantile of the Hausdorff distance of a subsample.

    Over n_sub random subsamples of size b of a point set with distance matrix D, the Hausdorff distance from the
    subsample to the set is computed and its (1 - alpha) quantile taken (Fasy et al. 2014, subsampling). With
    d_B <= 2 d_H (Lemma 1) and a bar of length L at distance L / 2 from the diagonal, bars shorter than 4 times that
    quantile are within the sampling noise of the diagram. The half-sample band of run_matched.py uses
    b = ceil(n_c / 2) and the matched-size band b = |P_c|, both with n_sub = 100 and alpha = 0.05.

    Args:
        D: (n, n) distance matrix of the point set.
        b: Subsample size.
        rng: numpy Generator.
        n_sub: Number of subsamples.
        alpha: The 1 - alpha quantile is used.

    Returns:
        The threshold as a float; 0.0 when b >= n (the subsample is the whole set).
    """
    n = len(D)
    if b >= n:
        return 0.0
    hd = [float(D[:, rng.choice(n, size=b, replace=False)].min(axis=1).max()) for _ in range(n_sub)]
    return 4.0 * float(np.quantile(hd, 1.0 - alpha))


def spot_class(pts, k):
    """SPOTgreedy on one class: k rows whose empirical distribution is closest in optimal transport to the class.

    Runs vendor.spotgreedy.SPOT_GreedySubsetSelection of the real-data arm on one class, with Euclidean cost and
    uniform weights, as baselines.spot_matched does.

    Args:
        pts: (m, d) coordinates of the class.
        k: Number of rows to select.

    Returns:
        Array of k local indices into pts.
    """
    sel, _ = SPOT_GreedySubsetSelection(cdist(pts, pts), np.ones(len(pts)) / len(pts), k)
    return sel


def longest(dgm):
    """Lifetime of the longest bar, 0 for an empty diagram.

    Args:
        dgm: (n, 2) array of finite bars (birth, death).

    Returns:
        The largest death - birth as a float, 0.0 when the diagram is empty.
    """
    return top_lifetimes(dgm, 1)[0] if len(dgm) else 0.0


def test_acc(X, y, S, Xt, yt):
    """Balanced accuracy of 1-NN on a selection of the training set, measured on a test set.

    Args:
        X: (n, d) training coordinates.
        y: (n,) training labels.
        S: Row indices of the selected training points.
        Xt: (m, d) test coordinates.
        yt: (m,) test labels.

    Returns:
        sklearn balanced accuracy of KNeighborsClassifier(n_neighbors=1) fitted on X[S], y[S], as a float.
    """
    pred = KNeighborsClassifier(n_neighbors=1).fit(X[S], y[S]).predict(Xt)
    return float(balanced_accuracy_score(yt, pred))


def task(args):
    """The matched-size comparison on one generator and seed: PENS against farthest-point, SPOTgreedy and random.

    For each (eps_scale, completion) of CONFIGS (eps_scale in {1, 4, 16}, completion consistent or gain), PENS is
    fitted at its other defaults and its per-class prototype counts |P_c| are handed to farthest-point sampling
    (topology.fps, a matched-size eps-net), SPOTgreedy (spot_class) and random selection (R_RANDOM = 5 draws,
    rng = default_rng([seed, class, config index, draw])). One row per class c and method records dB*_SX, the
    bottleneck distance between the diagram of the class-c selection and Dgm(X_c) in degrees 0, 1 and, for classes
    with beta2 > 0, 2 (the mean over draws for random), and for PENS also dB*_PK, the distance to the kept set K_c,
    next to bound = 2 eps_c. Every row counts the bars longer than three noise thresholds (band_threshold) in
    Dgm(X_c) (nlong*_X), in the selection (nlong*_S, the mean over draws) and, for PENS, in Dgm(K_c) (nlong*_K), with
    nlong*_match the share of draws whose count equals that of X_c: 4 eps_c (columns without suffix; four_eps), the
    length above which the certificate guarantees that a bar of Dgm(K_c) survives in Dgm(P_c) (Corollary 2); the
    half-sample band (suffix _half; thr_half, subsamples of size ceil(n_c / 2), rng default_rng([seed, c, 100]));
    and the matched-size band (suffix _matched; thr_matched, subsamples of size |P_c|, rng
    default_rng([seed, c, 200, config index])). longest*_{X,S,K} is the longest lifetime (mean over draws for S) and
    keep*{suffix} the share of draws whose longest bar exceeds the threshold. test_bal_acc is the balanced accuracy
    of 1-NN on the full selection (both classes) on a fresh draw of the generator at seed + 1000, as in
    run_ablation.py (mean over draws for random). Diagrams come from topology.diagrams through
    topology.DiagramCache. For consistent completion the farthest-point distances are asserted equal (relative
    tolerance TOL) to dB*_FX of results/certificate.csv for the same generator, seed, eps_scale and class, and the
    prototype counts to its n_proto.

    Args:
        args: Tuple (generator name, seed).

    Returns:
        List of row dicts with the keys COLS, one per (configuration, class, method).
    """
    name, seed = args
    X, y, truth = GENERATORS[name](seed)
    Xt, yt, _ = GENERATORS[name](seed + 1000)
    cert = pd.read_csv(RESULTS / "certificate.csv")
    cert = cert[(cert.gen == name) & (cert.seed == seed) & (cert.completion == "consistent")]
    dg = DiagramCache(X)
    idx_c = {c: np.flatnonzero(y == c) for c in (0, 1)}
    D_c = {c: cdist(X[idx_c[c]], X[idx_c[c]]) for c in (0, 1)}
    half = {c: band_threshold(D_c[c], (len(idx_c[c]) + 1) // 2, np.random.default_rng([seed, c, 100])) for c in (0, 1)}
    rows = []
    for ci, (lam, completion) in enumerate(CONFIGS):
        m = PENS(eps_scale=lam, completion=completion).fit(X, y)
        counts = {c: len(m.prototypes_[c]) for c in (0, 1)}
        selections = {
            "pens": [{c: np.asarray(m.prototypes_[c], dtype=int) for c in (0, 1)}],
            "fps": [{c: idx_c[c][fps(X[idx_c[c]], counts[c])] for c in (0, 1)}],
            "spot": [{c: idx_c[c][spot_class(X[idx_c[c]], counts[c])] for c in (0, 1)}],
            "random": [{c: np.random.default_rng([seed, c, ci, d]).choice(idx_c[c], size=counts[c], replace=False)
                        for c in (0, 1)} for d in range(R_RANDOM)],
        }
        acc = {meth: float(np.mean([test_acc(X, y, np.concatenate([s[0], s[1]]), Xt, yt) for s in sels]))
               for meth, sels in selections.items()}
        for c in (0, 1):
            dim = maxdim(truth[c])
            K, eps = np.asarray(m.kept_[c], dtype=int), float(m.radii_[c])
            dX, dK = dg(idx_c[c], dim), dg(K, dim)
            thr = {"eps": 4 * eps, "half": half[c],
                   "matched": band_threshold(D_c[c], counts[c], np.random.default_rng([seed, c, 200, ci]))}
            base = dict.fromkeys(COLS, NAN)
            base.update(gen=name, seed=seed, eps_scale=lam, completion=completion, cls=c, beta1_true=truth[c][1],
                        beta2_true=truth[c][2], maxdim=dim, n_c=len(idx_c[c]), n_kept=len(K), n_sel=counts[c],
                        eps=eps, bound=2 * eps, four_eps=thr["eps"], thr_half=thr["half"], thr_matched=thr["matched"])
            for k in range(dim + 1):
                base[f"longest{k}_X"] = longest(dX[k])
                for t, sfx in THRESHOLDS.items():
                    base[f"nlong{k}_X{sfx}"] = n_longer(dX[k], thr[t])
            for meth, sels in selections.items():
                dS = [dg(s[c], dim) for s in sels]
                row = dict(base, method=meth, test_bal_acc=acc[meth])
                for k in range(dim + 1):
                    row[f"dB{k}_SX"] = float(np.mean([bott(d[k], dX[k]) for d in dS]))
                    tops = [longest(d[k]) for d in dS]
                    row[f"longest{k}_S"] = float(np.mean(tops))
                    if meth == "pens":
                        row[f"dB{k}_PK"] = bott(dS[0][k], dK[k])
                        row[f"longest{k}_K"] = longest(dK[k])
                    for t, sfx in THRESHOLDS.items():
                        long = [n_longer(d[k], thr[t]) for d in dS]
                        row[f"nlong{k}_S{sfx}"] = float(np.mean(long))
                        row[f"nlong{k}_match{sfx}"] = float(np.mean([n == base[f"nlong{k}_X{sfx}"] for n in long]))
                        row[f"keep{k}{sfx}"] = float(np.mean([L > thr[t] for L in tops]))
                        if meth == "pens":
                            row[f"nlong{k}_K{sfx}"] = n_longer(dK[k], thr[t])
                rows.append(row)
                if meth == "fps" and completion == "consistent":
                    ref = cert[(cert.eps_scale == lam) & (cert.cls == c)]
                    assert len(ref) == 1, (name, seed, lam, c)
                    ref = ref.iloc[0]
                    assert int(ref.n_proto) == counts[c], (name, seed, lam, c, int(ref.n_proto), counts[c])
                    for k in range(dim + 1):
                        v = float(ref[f"dB{k}_FX"])
                        if np.isfinite(v):
                            assert abs(row[f"dB{k}_SX"] - v) <= TOL * max(1.0, abs(v)), (name, seed, lam, c, k,
                                                                                           row[f"dB{k}_SX"], v)
    return rows


def full_test_acc(name, seed):
    """Balanced accuracy of 1-NN on the whole training draw, on the fresh draw: the reference for the selections.

    Args:
        name: Generator name.
        seed: Training seed; the test draw is seed + 1000.

    Returns:
        The balanced accuracy as a float.
    """
    X, y, _ = GENERATORS[name](seed)
    Xt, yt, _ = GENERATORS[name](seed + 1000)
    return test_acc(X, y, np.arange(len(X)), Xt, yt)


def fmt(df, floats=4):
    """Render a DataFrame as text with a fixed number of decimals and blank cells for NaN.

    Args:
        df: pandas DataFrame.
        floats: Number of decimals.

    Returns:
        The string from df.to_string.
    """
    return df.to_string(float_format=lambda v: f"{v:.{floats}f}", na_rep="")


def gens_with(df, k):
    """Generators with a class whose diagrams reach degree k, in GENERATORS order.

    Args:
        df: DataFrame with the columns gen and maxdim.
        k: Homology degree.

    Returns:
        List of generator names.
    """
    present = set(df[df.maxdim >= k].gen)
    return [g for g in GENS if g in present]


def wilcoxon_lines(wide, what, higher_is_better=False):
    """Two-sided Wilcoxon signed-rank tests of PENS against each other method over the rows of `wide`, Holm-corrected.

    Args:
        wide: DataFrame indexed by generator with one column per method (per-generator medians or means).
        what: Label of the quantity, for the heading.
        higher_is_better: If True, differences are PENS minus method, otherwise method minus PENS, so that a
            positive difference and a win always mean PENS is better.

    Returns:
        List of text lines: the heading, one line per method with the mean difference, win-tie-loss counts, raw and
        Holm-adjusted p-values (corrected over the three methods; starred below ALPHA of the real-data arm's
        stats.py), and a footnote. The p-value is 1 when every difference is zero.
    """
    sign = "PENS minus method" if higher_is_better else "method minus PENS"
    lines = [f"{what}: Wilcoxon signed-rank (two-sided), PENS vs each method over the {len(wide)} per-generator "
             f"values, Holm-corrected over the three; differences are {sign}, so a positive difference and a win "
             "mean PENS is better.",
             f"{'Method':24s} {'Mean diff.':>10s} {'W-T-L':>9s} {'p':>9s} {'Holm p':>9s}"]
    rows = []
    for meth in METHODS[1:]:
        d = wide["pens"] - wide[meth] if higher_is_better else wide[meth] - wide["pens"]
        p = 1.0 if np.allclose(d, 0.0) else float(wilcoxon(wide["pens"], wide[meth]).pvalue)
        rows.append((meth, d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p))
    for (meth, diff, w, t, l, p), a in zip(rows, holm([r[-1] for r in rows])):
        flag = " *" if a < ALPHA else ""
        lines.append(f"{NAMES[meth]:24s} {diff:+10.4f} {f'{w}-{t}-{l}':>9s} {p:9.4f} {a:9.4f}{flag}")
    return lines + [f"* Holm-adjusted p < {ALPHA}", ""]


def long_bar_lines(df, p):
    """The long-bar block of the summary: thresholds, then match share and count difference by method.

    Reports the medians of the three thresholds (and their ratios to 4 eps_c, by eps_scale); then, per threshold
    and degree, the mean number of long bars in Dgm(X_c) and Dgm(K_c) and per method the share of class rows whose
    selection has exactly as many as X_c (match) and the mean absolute count difference, overall and by eps_scale;
    and a per-generator table of the match share at the half-sample band in degrees 1 and 2.

    Args:
        df: The matched rows of every method, restricted to class rows with eps_c > 0.
        p: The PENS rows of df, indexed by KEY.

    Returns:
        List of text lines.
    """
    lines = ["Long bars: bars with lifetime above a noise threshold, the same threshold for the four methods of a "
             "class row. Three thresholds: 4 eps_c, at which the certificate guarantees that a bar of Dgm(K_c) "
             f"survives in Dgm(P_c); the half-sample band, 4 times the {100 * (1 - BAND_ALPHA):g}% quantile over "
             f"{N_SUB} random subsamples of X_c of size ceil(n_c / 2) of the Hausdorff distance from the subsample to "
             "X_c (Fasy et al. 2014, with the factor of Lemma 1), below which a bar is within the sampling noise of "
             "Dgm(X_c); and the matched-size band, the same with subsamples of size |P_c|. For each threshold and "
             "degree: the mean number of such bars in Dgm(X_c) and in Dgm(K_c), and per method the share of class "
             "rows whose selection has exactly as many as X_c (match) and the mean absolute difference in the count "
             "(|diff|), overall and by eps_scale. Rows with eps_c = 0 are left out.", "",
             f"Thresholds over the class rows (medians): 4 eps_c {p.four_eps.median():.4f}, half-sample band "
             f"{p.thr_half.median():.4f} ({(p.thr_half / p.four_eps).median():.3f} of 4 eps_c), matched-size band "
             f"{p.thr_matched.median():.4f} ({(p.thr_matched / p.four_eps).median():.3f} of 4 eps_c); by eps_scale, "
             "the median ratio to 4 eps_c of the half-sample band is "
             + ", ".join(f"{lam:g}: {(p.thr_half / p.four_eps).groupby(level='eps_scale').median()[lam]:.3f}"
                         for lam in LAMBDAS) + " and of the matched-size band "
             + ", ".join(f"{lam:g}: {(p.thr_matched / p.four_eps).groupby(level='eps_scale').median()[lam]:.3f}"
                         for lam in LAMBDAS) + ".", ""]
    for t, sfx in THRESHOLDS.items():
        lines.append(f"-- {THRESHOLD_NAMES[t]} --")
        for k in DEG:
            rows_k = p[p.maxdim >= k]
            if rows_k.empty:
                continue
            lines.append(f"Degree {k} ({len(rows_k)} rows): mean bars in X_c {rows_k[f'nlong{k}_X{sfx}'].mean():.3f}, "
                         f"in K_c {rows_k[f'nlong{k}_K{sfx}'].mean():.3f}; X_c has at least one in "
                         f"{(rows_k[f'nlong{k}_X{sfx}'] >= 1).mean():.3f} of rows.")
            for meth in METHODS:
                q = df[(df.method == meth) & (df.maxdim >= k)].set_index(KEY).reindex(rows_k.index)
                diff = (q[f"nlong{k}_S{sfx}"] - rows_k[f"nlong{k}_X{sfx}"]).abs()
                by_l = q[f"nlong{k}_match{sfx}"].groupby(level="eps_scale").mean()
                lines.append(f"  {NAMES[meth]:14s} match {q[f'nlong{k}_match{sfx}'].mean():.3f}, |diff| "
                             f"{diff.mean():.3f} (" + ", ".join(f"eps_scale {lam:g}: {by_l[lam]:.3f}"
                                                                for lam in LAMBDAS) + ")")
        lines.append("")
    tables = {}
    for k in (1, 2):
        sub = df[df.maxdim >= k]
        t = sub.pivot_table(f"nlong{k}_match_half", "gen", "method", aggfunc="mean")
        tables[f"degree {k}"] = t.reindex(index=gens_with(df, k), columns=METHODS).rename(columns=NAMES)
    lines += ["Per generator, half-sample band: share of class rows (eps_c > 0) whose selection has exactly as many "
              "long bars as X_c, in degrees 1 and 2 (degree 2 where beta2 > 0), by method.",
              fmt(pd.concat(tables, axis=1), 3), ""]
    return lines


def longest_bar_lines(df, p):
    """The longest-bar block of the summary: how much of the longest bar of X_c each method retains.

    Per degree, the median over class rows with a bar in X_c of the ratio of the selection's longest bar to that of
    X_c, by method; then, per threshold, among the class rows whose longest bar of X_c exceeds the threshold, the
    share of rows (draws) in which the selection's longest bar exceeds it too, by method, overall and by eps_scale.

    Args:
        df: The matched rows of every method, restricted to class rows with eps_c > 0.
        p: The PENS rows of df, indexed by KEY.

    Returns:
        List of text lines.
    """
    lines = ["Longest bar: the lifetime of the longest bar of Dgm_k(X_c) and of the selection (mean over draws). Per "
             "degree: the median over class rows with a bar in X_c of the ratio of the selection's longest bar to "
             "that of X_c, by method; then, per threshold, among the class rows whose longest bar of X_c exceeds the "
             "threshold (a dominant feature above the noise), the share of rows (draws) in which the selection's "
             "longest bar exceeds it too, by method, overall and by eps_scale. Rows with eps_c = 0 are left out.", ""]
    for k in DEG:
        has = p[(p.maxdim >= k) & (p[f"longest{k}_X"] > 0)]
        if has.empty:
            continue
        parts = []
        for meth in METHODS:
            q = df[df.method == meth].set_index(KEY).reindex(has.index)
            parts.append(f"{NAMES[meth]} {float((q[f'longest{k}_S'] / has[f'longest{k}_X']).median()):.3f}")
        lines.append(f"Degree {k} ({len(has)} rows with a bar in X_c; median longest bar of X_c "
                     f"{has[f'longest{k}_X'].median():.4f}, of K_c {has[f'longest{k}_K'].median():.4f}): median ratio "
                     + ", ".join(parts) + ".")
    lines.append("")
    for t, sfx in THRESHOLDS.items():
        thr_col = {"eps": "four_eps", "half": "thr_half", "matched": "thr_matched"}[t]
        lines.append(f"-- {THRESHOLD_NAMES[t]} --")
        for k in DEG:
            rows_k = p[p.maxdim >= k]
            above = rows_k[rows_k[f"longest{k}_X"] > rows_k[thr_col]]
            if above.empty:
                lines.append(f"Degree {k}: no class row has its longest bar above the threshold.")
                continue
            parts = []
            for meth in METHODS:
                q = df[df.method == meth].set_index(KEY).reindex(above.index)
                by_l = q[f"keep{k}{sfx}"].groupby(level="eps_scale").mean()
                parts.append(f"{NAMES[meth]} {q[f'keep{k}{sfx}'].mean():.3f} ("
                             + ", ".join(f"{lam:g}: {by_l[lam]:.3f}" for lam in LAMBDAS if lam in by_l) + ")")
            lines.append(f"Degree {k} ({len(above)} of {len(rows_k)} rows above the threshold): kept above it by "
                         + "; ".join(parts) + ".")
        lines.append("")
    return lines


def summary(df):
    """Compose the text of results/matched.txt from the rows of results/matched.csv.

    Blocks, in order: per generator, the median d_B to X_c by degree and method and the mean balanced accuracy of
    1-NN on the full selection on the fresh draw, with the whole training draw as reference (full_test_acc); overall,
    the mean over generators of the per-generator medians and the share of class rows (generator x seed x eps_scale
    x completion x class) in which PENS is at least as close to X_c as each method, by completion and by eps_scale;
    Wilcoxon tests per degree and for balanced accuracy (wilcoxon_lines); the long-bar and longest-bar blocks on the
    rows with eps_c > 0 (long_bar_lines, longest_bar_lines); medians per generator and eps_scale; and what the
    certificate controls, the median of d_B(P_c, K_c) / (2 eps_c) by degree and the number of class rows in which
    the bound 2 eps_c holds (tolerance TOL).

    Args:
        df: DataFrame with the columns COLS.

    Returns:
        The summary text, newline-terminated.
    """
    lines = ["Matched-size topology comparison on the synthetic suite: PENS (eps_scale in {1, 4, 16}, consistent and "
             "gain completion) against a farthest-point net, SPOTgreedy and random selection (mean of 5 draws) at the "
             "same per-class sizes, on every generator and seeds 0-4. d_B is the bottleneck distance between the "
             "diagram of the class-c selection and the diagram of X_c; degree 2 only for classes with beta2 > 0. "
             "Medians are over seeds, eps_scale, completions and classes unless stated.", "",
             "Per generator: median d_B to X_c by degree and method, and the mean balanced accuracy of 1-NN on the "
             "full selection on a fresh clean draw of the generator (%).", ""]
    per_gen = {}
    for k in DEG:
        sub = df[df.maxdim >= k]
        wide = sub.pivot_table(f"dB{k}_SX", "gen", "method", aggfunc="median")
        per_gen[k] = wide.reindex(index=gens_with(df, k), columns=METHODS)
        lines += [f"Degree {k} (dB{k}_SX)", fmt(per_gen[k].rename(columns=NAMES)), ""]
    runs = df.drop_duplicates(["gen", "seed", "eps_scale", "completion", "method"])
    acc = 100 * runs.pivot_table("test_bal_acc", "gen", "method", aggfunc="mean").reindex(GENS)[METHODS]
    full = pd.Series({g: 100 * float(np.mean([full_test_acc(g, s) for s in SEEDS])) for g in GENS})
    lines += ["Balanced accuracy (%). The last column is the whole training draw on the same fresh draw, for "
              "reference.", fmt(acc.rename(columns=NAMES).assign(**{"Full training set": full}), 1), "",
              "Balanced accuracy, mean over generators (%): "
              + ", ".join(f"{NAMES[m]} {acc[m].mean():.1f}" for m in METHODS), ""]

    lines += ["Overall: mean over generators of the per-generator medians, and the share of class rows (generator x "
              "seed x eps_scale x completion x class) in which PENS is at least as close to X_c as the method, "
              "overall, by completion and by eps_scale.", ""]
    p = df[df.method == "pens"].set_index(KEY)
    for k in DEG:
        if per_gen[k].empty:
            continue
        means = ", ".join(f"{NAMES[m]} {per_gen[k][m].mean():.4f}" for m in METHODS)
        lines.append(f"Degree {k} ({len(per_gen[k])} generators): mean median {means}.")
        for meth in METHODS[1:]:
            q = df[df.method == meth].set_index(KEY)[f"dB{k}_SX"].reindex(p.index)
            ok = (p[f"dB{k}_SX"] <= q + 1e-12)[p[f"dB{k}_SX"].notna()]
            by_c = ok.groupby(level="completion").mean()
            by_l = ok.groupby(level="eps_scale").mean()
            lines.append(f"  PENS at least as close as {NAMES[meth]} in {ok.mean():.3f} of {len(ok)} rows "
                         f"(consistent {by_c['consistent']:.3f}, gain {by_c['gain']:.3f}; "
                         + ", ".join(f"eps_scale {lam:g}: {by_l[lam]:.3f}" for lam in LAMBDAS) + ").")
        lines.append("")
    for k in DEG:
        if not per_gen[k].empty:
            lines += wilcoxon_lines(per_gen[k], f"Degree {k}")
    lines += wilcoxon_lines(acc, "Balanced accuracy (points)", higher_is_better=True)

    sub = df[df.eps > 0]
    pk = sub[sub.method == "pens"].set_index(KEY)
    lines += long_bar_lines(sub, pk) + longest_bar_lines(sub, pk)

    lines += ["Per generator and eps_scale: median d_B to X_c by method (over seeds, completions and classes), and the "
              "median balanced accuracy (%).", ""]
    for k in DEG:
        sub_k = df[df.maxdim >= k]
        t = sub_k.pivot_table(f"dB{k}_SX", ["gen", "eps_scale"], "method", aggfunc="median")
        t = t.reindex(index=[(g, lam) for g in gens_with(df, k) for lam in LAMBDAS], columns=METHODS)
        lines += [f"Degree {k} (dB{k}_SX)", fmt(t.rename(columns=NAMES)), ""]
    t = 100 * runs.pivot_table("test_bal_acc", ["gen", "eps_scale"], "method", aggfunc="median")
    lines += ["Balanced accuracy (%)", fmt(t.reindex([(g, lam) for g in GENS for lam in LAMBDAS])[METHODS]
                                          .rename(columns=NAMES), 1), ""]

    ratio = pd.DataFrame({f"dB{k}_PK / 2eps": pk[f"dB{k}_PK"] / pk.bound for k in DEG})
    t = ratio.groupby(level=["gen", "eps_scale"]).median()
    worst = pk[[f"dB{k}_PK" for k in DEG]].fillna(0.0).max(axis=1)
    lines += ["What the certificate controls: median over seeds, completions and classes of d_B(P_c, K_c) / "
              "(2 eps_c) by degree (rows with eps_c > 0).",
              fmt(t.reindex([(g, lam) for g in GENS for lam in LAMBDAS])), "",
              f"The bound holds in {int((worst <= pk.bound + TOL).sum())} of {len(pk)} class rows with eps_c > 0."]
    return "\n".join(lines) + "\n"


def main():
    """Run the matched-size comparison on every generator and seed (0-4) in parallel, or only rebuild the summary.

    run_certificate.py already compares the prototypes of PENS with random subsets and a farthest-point net of X_c
    of the same size (dB*_RX and dB*_FX in results/certificate.csv); this script adds SPOTgreedy and the balanced
    accuracy of 1-NN on each selection, and summarizes the comparison per generator and overall (task, summary).
    Run after run_certificate.py (results/certificate.csv is read for a cross-check), from this directory:
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_matched.py
        python run_matched.py summary      (rewrites results/matched.txt from results/matched.csv, no fits)
    NPROC (default 4) sets the number of worker processes, one task per generator and seed and one thread each;
    generators with degree-2 classes are dispatched first. Imports stats and vendor.spotgreedy from ../real.

    Args:
        None, or the single word "summary" in sys.argv[1:] to skip the fits.

    Returns:
        None. Writes results/matched.csv (columns COLS, rows sorted by generator, seed, configuration, class and
        method) and results/matched.txt (the text of summary); prints the text and the total time.
    """
    if sys.argv[1:] == ["summary"]:
        text = summary(pd.read_csv(RESULTS / "matched.csv"))
        (RESULTS / "matched.txt").write_text(text)
        return print(text)
    RESULTS.mkdir(exist_ok=True)
    names = sorted(GENS, key=lambda g: -max(t[2] for t in GENERATORS[g](0)[2].values()))  # degree-2 classes first
    tasks = [(n, s) for n in names for s in SEEDS]
    t0 = time.time()
    rows = []
    with Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool:
        for out in pool.imap_unordered(task, tasks):
            rows.extend(out)
            print(f"{out[0]['gen']} seed {out[0]['seed']} done (elapsed {time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)[COLS]
    order = {m: i for i, m in enumerate(METHODS)}
    cfg = {c: i for i, c in enumerate(CONFIGS)}
    df = df.iloc[np.lexsort((df.method.map(order).to_numpy(), df.cls.to_numpy(),
                             [cfg[(lam, comp)] for lam, comp in zip(df.eps_scale, df.completion)],
                             df.seed.to_numpy(), df.gen.map(GENS.index).to_numpy()))].reset_index(drop=True)
    df.to_csv(RESULTS / "matched.csv", index=False)
    text = summary(df)
    (RESULTS / "matched.txt").write_text(text)
    print(text)
    print(f"TOTAL {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

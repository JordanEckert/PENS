import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DATASETS, IMBALANCED, RESULTS, pick_tradeoff
from run_components import GRID, LAMBDAS
from select_real import METRICS
from stats import ALPHA, holm

VARIANTS = ["noedit", "nocomp", "fps"]
REFERENCE = ["full", "random", "spot"]
NAMES = {"pens": "PENS", "noedit": "PENS without edit", "nocomp": "PENS without completion",
         "fps": "Farthest-point (size-matched)", "full": "Full training set", "random": "Random (matched)",
         "spot": "SPOTgreedy (matched)"}
TITLES = {"bal_acc": "Balanced accuracy", "reduction": "Reduction", "effectiveness": "Effectiveness",
          "time_s": "Fit time"}
COLUMNS = ["dataset", "fold", "method", "config", "n_train", "n_selected"] + METRICS
ORDER = {m: {label: i for i, (label, _) in enumerate(grid)} for m, grid in GRID.items()}
GROUPS = [("All (18)", DATASETS), ("Imbalanced (5)", IMBALANCED)]


def concat_shards():
    """Concatenate the shards of run_components.py and their refit records.

    Reads results/components_shards/<dataset>.csv and <dataset>_refit.csv for every dataset of DATASETS and checks
    the row counts: per fold, every tuned configuration of GRID, one "pens_selected" row per tuned variant and one
    fps row; one refit row per fold.

    Returns:
        Tuple (raw, refit): the concatenated result rows (also written to results/components_raw.csv) and the
        concatenated refit records (also written to results/components_refit.csv).

    Raises:
        AssertionError: If either row count differs from the expected one.
    """
    parts = [pd.read_csv(RESULTS / "components_shards" / f"{n}.csv", float_precision="round_trip") for n in DATASETS]
    raw = pd.concat(parts, ignore_index=True)
    per_fold = sum(len(g) for g in GRID.values()) + len(GRID) + 1  # tuned grids, pens_selected rows, fps
    assert len(raw) == 5 * len(DATASETS) * per_fold, (len(raw), 5 * len(DATASETS) * per_fold)
    raw.to_csv(RESULTS / "components_raw.csv", index=False)
    refit = pd.concat([pd.read_csv(RESULTS / "components_shards" / f"{n}_refit.csv") for n in DATASETS],
                      ignore_index=True)
    assert len(refit) == 5 * len(DATASETS), len(refit)
    refit.to_csv(RESULTS / "components_refit.csv", index=False)
    return raw, refit


def refit_lines(refit):
    """Lines of the refit check of run_components.py for the summary.

    Lists every (dataset, fold) where the PENS refit at the selected configuration does not reproduce n_selected of
    real.csv, with both values, and the datasets whose training parts carry exact ties (duplicate rows or repeated
    pairwise distances), on which such mismatches can arise across machines.

    Args:
        refit: Refit records from concat_shards (REFIT_COLUMNS of run_components.py).

    Returns:
        List of text lines, ending with an empty line.
    """
    bad = refit[refit.n_refit != refit.n_selected]
    tied = refit[(refit.duplicate_rows > 0) | (refit.tied_distances > 0)]
    tied_sets = sorted(set(tied.dataset), key=DATASETS.index)
    lines = [f"Refit check. The PENS refit at the configuration selected in real.csv reproduces n_selected on "
             f"{len(refit) - len(bad)} of {len(refit)} folds. "
             + ("No fold differs." if bad.empty else
                f"{len(bad)} folds differ (dataset, fold, configuration, real.csv, refit): "
                + "; ".join(f"{r.dataset} {r.fold} {r.config} {r.n_selected} {r.n_refit}" for r in bad.itertuples())
                + ". On those folds the farthest-point selection is matched to the refit."),
             f"Training parts with exact ties (duplicate rows or repeated pairwise distances) occur in "
             f"{len(tied_sets)} datasets ({', '.join(tied_sets) or 'none'}); "
             f"{len(set(bad.dataset) & set(tied_sets))} of them are affected by a mismatch, and "
             f"{len(set(bad.dataset) - set(tied_sets))} datasets without exact ties are.", ""]
    return lines


def select(raw, real):
    """Pick one configuration per (dataset, outer fold, variant) and join the PENS rows.

    noedit and nocomp are reduced by the trade-off rule on the inner-CV columns (the code path of select_real.select,
    configurations in GRID order so that ties go to grid order); the single fps row is kept; the PENS rows are copied
    from real.csv; effectiveness = balanced accuracy x reduction. The fps rows are checked to carry the PENS
    configuration, its n_selected and its reduction.

    Args:
        raw: Result rows from concat_shards.
        real: results/real.csv as a DataFrame.

    Returns:
        DataFrame with the columns COLUMNS, one row per dataset, fold and method (pens, noedit, nocomp, fps), sorted
        by dataset, fold and method.

    Raises:
        AssertionError: If a group does not reduce to one row, or an fps row does not match its PENS row.
    """
    rows = []
    tuned = raw[raw.config != "pens_selected"]
    for (name, fold, method), g in tuned.groupby(["dataset", "fold", "method"], sort=False):
        if method in ORDER:
            g = g.iloc[np.argsort(g.config.map(ORDER[method]).to_numpy(), kind="stable")]
            g = g.iloc[[pick_tradeoff(g.inner_bal_acc.to_numpy(), g.inner_reduction.to_numpy())]]
        assert len(g) == 1, (name, fold, method)
        rows.append(g.iloc[0])
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["effectiveness"] = df.bal_acc * df.reduction
    df = pd.concat([real[real.method == "pens"][COLUMNS], df[COLUMNS]], ignore_index=True)
    order = {m: i for i, m in enumerate(["pens"] + VARIANTS)}
    df = df.iloc[np.lexsort((df.method.map(order).to_numpy(), df.fold.to_numpy(),
                             df.dataset.map(DATASETS.index).to_numpy()))].reset_index(drop=True)
    pens = df[df.method == "pens"].set_index(["dataset", "fold"])
    fps = df[df.method == "fps"].set_index(["dataset", "fold"])
    assert (fps.config == pens.config).all() and (fps.n_selected == pens.n_selected).all()
    assert np.allclose(fps.reduction, pens.reduction)
    return df


def _row(name, r, extra=""):
    """Format one table line: method name, balanced accuracy, reduction and effectiveness in %, fit time in s.

    Args:
        name: Display name of the method.
        r: Row (Series) with bal_acc, reduction, effectiveness and time_s.
        extra: Text appended to the line (further columns).

    Returns:
        The formatted line.
    """
    return (f"{name:32s} {100 * r.bal_acc:9.1f} {100 * r.reduction:9.1f} {100 * r.effectiveness:9.1f} "
            f"{r.time_s:9.2f}{extra}")


def means_tables(per):
    """Table T-A1 (tuned) of the summary.

    Means over the 18 datasets, and over the 5 imbalanced ones, of the per-dataset fold means of balanced accuracy,
    reduction, effectiveness and fit time for pens, noedit, nocomp and fps, with the full training set, random
    (matched) and SPOTgreedy (matched) from real.csv for reference.

    Args:
        per: Per-(dataset, method) means of METRICS, including the REFERENCE methods.

    Returns:
        List of text lines.
    """
    lines = ["T-A1 (tuned). Means over datasets of the per-dataset means over 5 outer folds; balanced accuracy, "
             "reduction and effectiveness in %, fit time in seconds. Removing the edit also changes the radius, "
             "since eps = eps_scale x median merge height of K and K is then the whole training part. Fit times are "
             "comparable with results/summary.txt only if run_components.py ran on the machine of the reported "
             "run_real.py runs (the Mac Mini).", ""]
    for title, names in GROUPS:
        sub = per[per.dataset.isin(names)].groupby("method")[METRICS].mean()
        lines += [title, f"{'Method':32s} {'Bal. acc.':>9s} {'Reduction':>9s} {'Effect.':>9s} {'Time (s)':>9s}"]
        for m in ["pens"] + VARIANTS + REFERENCE:
            lines.append(_row(NAMES[m], sub.loc[m]))
        lines.append("")
    return lines


def wilcoxon_tables(per):
    """Wilcoxon tables of the summary, in the layout of stats.py.

    Two-sided Wilcoxon signed-rank tests of pens against each variant over the 18 dataset means of every metric in
    METRICS, with Holm's correction over the three variants within each metric and the datasets won-tied-lost.
    Differences are PENS minus variant (in points; in seconds for fit time, where a negative difference means PENS is
    faster); a comparison in which every dataset ties gets p = 1.

    Args:
        per: Per-(dataset, method) means of METRICS.

    Returns:
        List of text lines.
    """
    lines = [f"Wilcoxon signed-rank (two-sided), PENS vs each variant over N = {per.dataset.nunique()} datasets, "
             f"Holm-corrected over the three variants within each metric; alpha = {ALPHA}. Differences are PENS "
             "minus variant (in points; in seconds for fit time, where a negative difference means PENS is faster); "
             "W-T-L counts the datasets on which PENS is better, tied and worse.", ""]
    for metric in METRICS:
        wide = per.pivot(index="dataset", columns="method", values=metric)
        rows = []
        for m in VARIANTS:
            d = wide["pens"] - wide[m]
            better = -d if metric == "time_s" else d
            p = 1.0 if np.allclose(d, 0.0) else float(wilcoxon(wide["pens"], wide[m]).pvalue)
            rows.append((m, d.mean(), int((better > 0).sum()), int((better == 0).sum()), int((better < 0).sum()), p))
        adj = holm([r[-1] for r in rows])
        lines += [f"{TITLES[metric]}", f"{'Variant':32s} {'Mean diff.':>10s} {'W-T-L':>9s} {'p':>9s} {'Holm p':>9s}"]
        for (m, diff, w, t, l, p), a in zip(rows, adj):
            scale = 1.0 if metric == "time_s" else 100.0
            flag = " *" if a < ALPHA else ""
            lines.append(f"{NAMES[m]:32s} {scale * diff:+10.2f} {f'{w}-{t}-{l}':>9s} {p:9.4f} {a:9.4f}{flag}")
        lines.append("")
    return lines + [f"* Holm-adjusted p < {ALPHA}", ""]


def same_setting_tables(raw, df):
    """Table T-A2 (same setting) of the summary.

    noedit and nocomp at the configuration PENS selected on the fold (config "pens_selected"), with the same
    averaging as T-A1, plus the mean kept share |K| / n_train (from results/certificate_real.csv for PENS and nocomp,
    whose kept sets coincide, and 1 for noedit by construction) and the mean number of prototypes, next to PENS.

    Args:
        raw: Result rows from concat_shards (the "pens_selected" rows are used).
        df: Selected rows from select (the PENS rows are used).

    Returns:
        List of text lines.
    """
    cert = pd.read_csv(RESULTS / "certificate_real.csv")
    cert = cert[cert.selected].groupby(["dataset", "fold"]).agg(kept=("n_kept", "sum"), n=("n_c", "sum"))
    kept_share = (cert.kept / cert.n).rename("kept_share")
    same = raw[raw.config == "pens_selected"].copy()
    same["effectiveness"] = same.bal_acc * same.reduction
    rows = pd.concat([df[df.method == "pens"][COLUMNS], same[COLUMNS]], ignore_index=True)
    rows = rows.join(kept_share, on=["dataset", "fold"])
    rows.loc[rows.method == "noedit", "kept_share"] = 1.0
    per = rows.groupby(["dataset", "method"])[METRICS + ["kept_share", "n_selected"]].mean().reset_index()
    lines = ["T-A2 (same setting). noedit and nocomp at the eps_scale (and, for noedit, the completion) that PENS "
             "selected on the fold; same averaging as T-A1, plus the mean kept share |K| / n_train in % and the "
             "mean number of prototypes.", ""]
    for title, names in GROUPS:
        sub = per[per.dataset.isin(names)].groupby("method")[METRICS + ["kept_share", "n_selected"]].mean()
        lines += [title, f"{'Method':32s} {'Bal. acc.':>9s} {'Reduction':>9s} {'Effect.':>9s} {'Time (s)':>9s} "
                         f"{'Kept (%)':>9s} {'Protos':>9s}"]
        for m in ["pens", "noedit", "nocomp"]:
            r = sub.loc[m]
            lines.append(_row(NAMES[m], r, f" {100 * r.kept_share:9.1f} {r.n_selected:9.1f}"))
        lines.append("")
    return lines


def picks_tables(df):
    """Tables of how often the trade-off rule of each variant chose each eps_scale, and each completion for noedit.

    Args:
        df: Selected rows from select; the eps_scale and completion are parsed from the config labels of the pens,
            noedit and nocomp rows.

    Returns:
        List of text lines: a counts table by eps_scale (of 90 folds) for pens, noedit and nocomp, then a counts
        table by completion for pens and noedit.
    """
    sel = df[df.method.isin(["pens", "noedit", "nocomp"])].copy()
    sel["eps_scale"] = sel.config.str.extract(r"^l(\d+)")[0].astype(float)
    sel["completion"] = sel.config.str.extract(r"_(\w+)$")[0]
    lam = sel.pivot_table(index="eps_scale", columns="method", values="fold", aggfunc="count", fill_value=0)
    lam = lam.reindex(index=list(LAMBDAS), columns=["pens", "noedit", "nocomp"], fill_value=0)
    comp = sel[sel.method != "nocomp"].pivot_table(index="completion", columns="method", values="fold",
                                                   aggfunc="count", fill_value=0)
    comp = comp.reindex(index=["gain", "consistent"], columns=["pens", "noedit"], fill_value=0)
    return ["Configurations chosen by the trade-off rule (of 90 folds): eps_scale by variant, then completion.", "",
            lam.rename(columns=NAMES).to_string(), "", comp.rename(columns=NAMES).to_string(), ""]


def main():
    """Real-data arm, component comparison (A), phase 2: one configuration per variant and fold, and summary tables.

    Run after run_components.py has written every shard: python select_components.py (no arguments). Reads
    results/components_shards/<dataset>.csv and <dataset>_refit.csv for the 18 datasets, results/real.csv (the PENS
    rows and the reference methods full, random and spot) and results/certificate_real.csv (kept shares for T-A2).

    Returns:
        None. Writes results/components_raw.csv (all shards concatenated), results/components_refit.csv (the
        per-fold refit record), results/components.csv (one row per dataset, fold and variant, with effectiveness)
        and results/components_summary.txt, which is also printed and holds, in order: T-A1 (tuned means), the
        Wilcoxon tests, T-A2 (same setting), the configurations chosen by the trade-off rule and the refit check.
    """
    real = pd.read_csv(RESULTS / "real.csv")
    raw, refit = concat_shards()
    df = select(raw, real)
    df.to_csv(RESULTS / "components.csv", index=False)
    per = pd.concat([df, real[real.method.isin(REFERENCE)]]).groupby(["dataset", "method"], sort=False)[METRICS].mean()
    per = per.reset_index()
    lines = ["Real-data arm, component comparison (A): PENS against PENS without the edit (purity=1, s_min=1), "
             "without completion (completion=None), both tuned by inner CV and the trade-off rule like PENS, and "
             "against class-wise farthest-point selection at the per-class sizes PENS selected.", ""]
    lines += (means_tables(per) + wilcoxon_tables(per) + same_setting_tables(raw, df) + picks_tables(df)
              + refit_lines(refit))
    text = "\n".join(lines) + "\n"
    (RESULTS / "components_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

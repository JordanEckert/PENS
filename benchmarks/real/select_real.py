import numpy as np
import pandas as pd

from common import DATASETS, GRIDS, IMBALANCED, MATCHED, RESULTS, grid, load, pick_tradeoff

METHODS = ["pens", "full", "enn", "cnn", "drop3", "icf", "ib3", "hmnei", "lsbo", "ps", "gkn", "random", "spot"]
NAMES = {"pens": "PENS", "full": "Full training set", "enn": "ENN", "cnn": "CNN", "drop3": "DROP3", "icf": "ICF",
         "ib3": "IB3", "hmnei": "HMN-EI", "lsbo": "LSBo", "ps": "PS (Bien-Tibshirani)",
         "gkn": "GKN (Gottlieb et al.)", "random": "Random (matched)",
         "spot": "SPOTgreedy (matched)"}
METRICS = ["bal_acc", "reduction", "effectiveness", "time_s"]
COLUMNS = ["dataset", "fold", "method", "config", "n_train", "n_selected"] + METRICS
ORDER = {m: {label: i for i, (label, _) in enumerate(grid)} for m, grid in GRIDS.items()}


def concat_shards():
    """Concatenate the per-dataset shards of run_real.py into one raw table.

    Reads results/shards/<dataset>.csv for every dataset of DATASETS and checks that the row count equals 5 folds x
    (every configuration of every method in GRIDS, less the k values common.grid leaves out, plus the two
    size-matched baselines), which needs the datasets themselves.

    Returns:
        The concatenated DataFrame (the columns of run_real.COLUMNS), also written to results/real_raw.csv.

    Raises:
        AssertionError: If the number of rows differs from the expected count.
    """
    parts = [pd.read_csv(RESULTS / "shards" / f"{n}.csv", float_precision="round_trip") for n in DATASETS]
    raw = pd.concat(parts, ignore_index=True)
    expected = 0
    for n in DATASETS:
        X, y = load(n)
        expected += 5 * (sum(len(grid(m, X, y)) for m in GRIDS) + len(MATCHED))
    assert len(raw) == expected, (len(raw), expected)
    raw.to_csv(RESULTS / "real_raw.csv", index=False)
    return raw


def select(raw):
    """Pick one configuration per (dataset, outer fold, method) and add effectiveness.

    Methods with more than one configuration are reduced by the trade-off rule (common.pick_tradeoff) on the
    inner-CV columns, with the configurations in GRIDS order so that ties go to grid order; methods with a single
    configuration keep it. effectiveness = balanced accuracy x reduction. The matched baselines (random, spot) are
    checked to carry the PENS configuration that the same rule picked in run_real.py.

    Args:
        raw: DataFrame from concat_shards.

    Returns:
        DataFrame with the columns COLUMNS and one row per dataset, fold and method.

    Raises:
        AssertionError: If a group does not reduce to one row, or a matched baseline's config differs from PENS's.
    """
    rows = []
    for (name, fold, method), g in raw.groupby(["dataset", "fold", "method"], sort=False):
        if method in ORDER and len(g) > 1:
            g = g.iloc[np.argsort(g.config.map(ORDER[method]).to_numpy(), kind="stable")]
            g = g.iloc[[pick_tradeoff(g.inner_bal_acc.to_numpy(), g.inner_reduction.to_numpy())]]
        assert len(g) == 1, (name, fold, method)
        rows.append(g.iloc[0])
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["effectiveness"] = df.bal_acc * df.reduction
    # The matched baselines were fitted at the PENS configuration that the same rule picked in run_real.py.
    pens = df[df.method == "pens"].set_index(["dataset", "fold"]).config
    for m in MATCHED:
        assert (df[df.method == m].set_index(["dataset", "fold"]).config == pens).all(), m
    return df[COLUMNS]


def summary(df, per_dataset):
    """Format the text of results/summary.txt.

    Tables of mean balanced accuracy, reduction and effectiveness (in %) and fit time (in s), sorted by
    effectiveness, over all 18 datasets, the 13 balanced ones and the 5 imbalanced ones; a per-dataset table of
    every metric; and the configurations the trade-off rule picked for PENS and for every tuned competitor.

    Args:
        df: Selected rows from select (one per dataset, fold and method).
        per_dataset: Means of METRICS over the five folds per (dataset, method).

    Returns:
        The summary text, ending with a newline.
    """
    lines = ["Real-data arm: means over datasets of the per-dataset means over 5 outer folds.",
             "Balanced accuracy, reduction and effectiveness in %, fit time in seconds.", ""]
    groups = [("All (18)", DATASETS), ("Balanced (13)", [d for d in DATASETS if d not in IMBALANCED]),
              ("Imbalanced (5)", IMBALANCED)]
    for title, names in groups:
        sub = per_dataset[per_dataset.dataset.isin(names)].groupby("method")[METRICS].mean()
        lines += [f"{title}", f"{'Method':24s} {'Bal. acc.':>9s} {'Reduction':>9s} {'Effect.':>9s} {'Time (s)':>9s}"]
        for m in sub.sort_values("effectiveness", ascending=False).index:
            r = sub.loc[m]
            lines.append(f"{NAMES[m]:24s} {100 * r.bal_acc:9.1f} {100 * r.reduction:9.1f} "
                         f"{100 * r.effectiveness:9.1f} {r.time_s:9.2f}")
        lines.append("")
    for metric, scale in [("bal_acc", 100), ("reduction", 100), ("effectiveness", 100), ("time_s", 1)]:
        wide = per_dataset.pivot(index="dataset", columns="method", values=metric)[METHODS] * scale
        lines += [f"Per dataset: {metric}" + (" (%)" if scale == 100 else " (s)"),
                  wide.rename(columns=NAMES).to_string(float_format=lambda v: f"{v:.1f}" if scale == 100 else f"{v:.2f}"), ""]
    picks = df[df.method == "pens"].config.value_counts()
    lines += ["PENS configurations picked by the trade-off rule (of 90 folds): "
              + ", ".join(f"{c} {n}" for c, n in picks.items())]
    for m in ["enn", "cnn", "drop3", "icf", "ib3", "hmnei", "ps", "gkn"]:
        picks = df[df.method == m].config.value_counts()
        lines.append(f"{NAMES[m]}: " + ", ".join(f"{c} {n}" for c, n in picks.items()))
    return "\n".join(lines) + "\n"


def main():
    """Real-data arm, phase 2: one configuration per method and outer fold, and summary tables.

    Run after run_real.py has written every shard: python select_real.py (no arguments). Reads
    results/shards/<dataset>.csv for the 18 datasets of common.DATASETS, and the datasets (to recount the expected
    rows).

    Returns:
        None. Writes results/real_raw.csv (all shards concatenated), results/real.csv (one row per dataset, fold and
        method, with effectiveness), results/per_dataset.csv (means over the five folds) and results/summary.txt,
        which is also printed.
    """
    df = select(concat_shards())
    df.to_csv(RESULTS / "real.csv", index=False)
    per_dataset = df.groupby(["dataset", "method"], sort=False)[METRICS].mean().reset_index()
    per_dataset.to_csv(RESULTS / "per_dataset.csv", index=False)
    text = summary(df, per_dataset)
    (RESULTS / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

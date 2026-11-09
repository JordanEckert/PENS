import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon  # noqa: E402

from common import RESULTS  # noqa: E402
from select_real import METHODS, NAMES  # noqa: E402

ALPHA = 0.05
TESTED = [m for m in METHODS if m != "full"]
COMPETITORS = [m for m in TESTED if m != "pens"]
# (column, title, higher is better)
METRICS = [("bal_acc", "Balanced accuracy", True), ("reduction", "Reduction", True),
           ("effectiveness", "Effectiveness", True), ("time_s", "Fit time", False)]
# The size-matched baselines, starred in the cd_<metric>_ms figures the manuscript uses.
MATCHED = ("random", "spot")
MATCHED_STARRED = ("bal_acc", "effectiveness")
INK, INK_2, ACCENT = "#0b0b0b", "#52514e", "#2a78d6"


def holm(p):
    """Holm-adjusted p-values (step-down Bonferroni, monotone, capped at 1).

    Args:
        p: Sequence of raw p-values.

    Returns:
        Float array of adjusted p-values in the order of the input.
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def wilcoxon_table(wide, metric, title):
    """Wilcoxon signed-rank tests (two-sided) of PENS against every competitor on one metric, as text lines.

    Holm's correction is applied over the competitors within the metric. A comparison in which every dataset ties
    (reduction against the size-matched baselines) gets p = 1. Differences are PENS minus competitor, reported in
    points together with the datasets won-tied-lost.

    Args:
        wide: DataFrame indexed by dataset with one column per method of TESTED (per-dataset means of the metric).
        metric: Column name of the metric (unused in the text; kept for the call site).
        title: Metric title for the header line.

    Returns:
        List of text lines, ending with the asterisk legend and an empty line.
    """
    rows = []
    for m in COMPETITORS:
        d = wide["pens"] - wide[m]
        p = 1.0 if np.allclose(d, 0.0) else float(wilcoxon(wide["pens"], wide[m]).pvalue)
        rows.append((m, d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p))
    adj = holm([r[-1] for r in rows])
    lines = [f"Wilcoxon signed-rank, PENS vs competitor: {title} (mean difference in points, datasets won-tied-lost)",
             f"{'Competitor':24s} {'Mean diff.':>10s} {'W-T-L':>9s} {'p':>9s} {'Holm p':>9s}"]
    for (m, diff, w, t, l, p), a in zip(rows, adj):
        flag = " *" if a < ALPHA else ""
        lines.append(f"{NAMES[m]:24s} {100 * diff:+10.2f} {f'{w}-{t}-{l}':>9s} {p:9.4f} {a:9.4f}{flag}")
    return lines + [f"* Holm-adjusted p < {ALPHA}", ""]


def cliques(avg, cd):
    """Maximal groups of methods (in rank order) whose average ranks differ by less than the CD.

    Args:
        avg: Average ranks sorted in increasing order.
        cd: Nemenyi critical difference.

    Returns:
        List of (first, last) index pairs into avg, inclusive, one per group of at least two methods; a group
        contained in an earlier one is not repeated.
    """
    out = []
    for i in range(len(avg)):
        j = i
        while j + 1 < len(avg) and avg[j + 1] - avg[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


def cd_diagram(names, avg, cd, title, subtitle, path):
    """Critical-difference diagram in the layout of Demsar (2006): rank axis on top, best rank on the left.

    Methods joined by a bar are not significantly different (Nemenyi); PENS is drawn in the accent color and bold.
    The CD bar is drawn above the worst end of the axis.

    Args:
        names: Display names of the methods, in increasing order of average rank.
        avg: Their average ranks, sorted increasingly.
        cd: Nemenyi critical difference.
        title: Figure title.
        subtitle: List of subtitle lines.
        path: Output path without suffix; .pdf and .png (200 dpi, white background) are written.

    Returns:
        None.
    """
    k = len(names)
    half = (k + 1) // 2
    lo, hi = 1, k
    groups = cliques(avg, cd)
    width, row = 9.0, 0.32
    y_axis = 1.45
    y_label0 = y_axis + 0.3 + 0.12 * len(groups)
    fig_h = y_label0 + row * (half - 1) + 0.25
    fig, ax = plt.subplots(figsize=(width, fig_h))
    ax.set_xlim(0, width)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    x0, x1 = 2.5, width - 2.5

    def xpos(r):
        """Map an average rank r in [lo, hi] linearly to its x coordinate on the rank axis [x0, x1]."""
        return x0 + (r - lo) / (hi - lo) * (x1 - x0)

    ax.text(0, 0.05, title, fontsize=11, color=INK, fontweight="bold", va="top")
    for i, line in enumerate(subtitle):
        ax.text(0, 0.35 + 0.22 * i, line, fontsize=8.5, color=INK_2, va="top")
    ax.plot([xpos(lo), xpos(hi)], [y_axis, y_axis], color=INK_2, lw=1)
    for r in range(lo, hi + 1):
        ax.plot([xpos(r)] * 2, [y_axis - 0.08, y_axis], color=INK_2, lw=1)
        ax.text(xpos(r), y_axis - 0.12, str(r), ha="center", va="bottom", fontsize=8, color=INK_2)
    # CD bar, right-aligned above the worst end of the axis.
    y_cd = y_axis - 0.42
    ax.plot([xpos(hi - cd), xpos(hi)], [y_cd, y_cd], color=INK, lw=1.5)
    for r in (hi - cd, hi):
        ax.plot([xpos(r)] * 2, [y_cd - 0.05, y_cd + 0.05], color=INK, lw=1.5)
    ax.text(xpos(hi - cd / 2), y_cd - 0.07, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

    for i, (name, r) in enumerate(zip(names, avg)):
        left = i < half
        y = y_label0 + row * (i if left else k - 1 - i)
        xe = x0 - 0.15 if left else x1 + 0.15
        color = ACCENT if name == "PENS" else INK
        lw = 2.0 if name == "PENS" else 1.0
        ax.plot([xpos(r), xpos(r), xe], [y_axis, y, y], color=color, lw=lw)
        ax.text(xe + (-0.08 if left else 0.08), y, f"{name} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=8.5, color=INK, fontweight="bold" if name == "PENS" else "normal")
    for g, (i, j) in enumerate(groups):
        y = y_axis + 0.18 + 0.12 * g
        ax.plot([xpos(avg[i]) - 0.04, xpos(avg[j]) + 0.04], [y, y], color=INK, lw=3, solid_capstyle="round")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    """Statistical comparison of the methods over the 18 datasets of the real-data arm.

    Run after select_real.py: python stats.py (no arguments). Reads results/per_dataset.csv, in which each metric is
    already averaged over the five outer folds of a dataset. The full training set is a reference: it is excluded
    from the tests, since its reduction and effectiveness are 0 by definition.
    1. Wilcoxon signed-rank tests (two-sided) of PENS against every competitor, for balanced accuracy, reduction
       and effectiveness, with a Holm correction over the competitors within each metric (wilcoxon_table).
    2. For each of balanced accuracy, reduction, effectiveness and fit time, the Friedman test over all methods and
       the Nemenyi critical difference (Demsar 2006), CD = q_alpha * sqrt(k (k + 1) / (6 N)), with q_alpha the
       studentized range quantile over sqrt(2); average ranks and the groups that do not differ significantly are
       listed, and each metric gets a critical-difference diagram (cd_diagram).

    Returns:
        None. Writes results/stats.txt (also printed) and results/cd_<metric>.pdf and .png for the four metrics; for
        balanced accuracy and effectiveness also results/cd_<metric>_ms.pdf and .png, the manuscript's figures, in
        which an asterisk marks the size-matched baselines.
    """
    per = pd.read_csv(RESULTS / "per_dataset.csv")
    n_data = per.dataset.nunique()
    lines = [f"Statistical tests over N = {n_data} datasets (per-dataset means over 5 outer folds); "
             f"alpha = {ALPHA}. The full training set is excluded.", ""]
    for metric, title, _ in METRICS[:3]:
        wide = per.pivot(index="dataset", columns="method", values=metric)[TESTED]
        lines += wilcoxon_table(wide, metric, title)
    k = len(TESTED)
    q = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_data))
    lines += [f"Friedman tests and Nemenyi critical difference: k = {k} methods, q_{ALPHA} = {q:.3f}, CD = {cd:.3f}", ""]
    for metric, title, higher in METRICS:
        wide = per.pivot(index="dataset", columns="method", values=metric)[TESTED]
        stat, p = friedmanchisquare(*[wide[m].to_numpy() for m in TESTED])
        ranks = np.vstack([rankdata(-row if higher else row) for row in wide.to_numpy()])
        avg = pd.Series(ranks.mean(axis=0), index=TESTED).sort_values(kind="stable")
        names = [NAMES[m] for m in avg.index]
        groups = cliques(avg.to_numpy(), cd)
        lines += [f"{title}: Friedman chi2 = {stat:.2f} (df = {k - 1}), p = {p:.2e}",
                  "Average ranks (1 = best): " + ", ".join(f"{n} {r:.2f}" for n, r in zip(names, avg)),
                  "Groups not significantly different (Nemenyi): "
                  + "; ".join("{" + ", ".join(names[i:j + 1]) + "}" for i, j in groups), ""]
        subtitle = [f"Average rank over {n_data} datasets (1 = best); Friedman p = {p:.1e}.",
                    f"Methods joined by a bar do not differ significantly (Nemenyi, alpha = {ALPHA})."]
        cd_diagram(names, avg.to_numpy(), cd, title, subtitle, RESULTS / f"cd_{metric}")
        if metric in MATCHED_STARRED:
            # The manuscript's version: an asterisk marks the size-matched baselines and nothing else changes.
            starred = [NAMES[m] + ("*" if m in MATCHED else "") for m in avg.index]
            cd_diagram(starred, avg.to_numpy(), cd, title, subtitle, RESULTS / f"cd_{metric}_ms")
    text = "\n".join(lines) + "\n"
    (RESULTS / "stats.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

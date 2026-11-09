import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import GRIDS, HERE, RESULTS  # noqa: E402

FIGURES = HERE / "figures"
CONFIGS = dict(GRIDS["pens"])
LAMBDAS = (1.0, 2.0, 4.0, 8.0, 16.0)
COMPLETIONS = ("consistent", "gain")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
STYLE = {"consistent": "-", "gain": "--"}
LAMBDA = r"$\lambda$"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Per-configuration means (balanced accuracy %, reduction %) and the share of folds with at least one certified bar
# by eps_scale (degree 0, degree 1) from certificate_real.txt, as checked on 2026-10-23.
MEANS = {"l1_gain": (80.2, 37.6), "l1_consistent": (80.3, 36.9), "l2_gain": (78.5, 78.4), "l2_consistent": (79.5, 69.3),
         "l4_gain": (77.8, 92.4), "l4_consistent": (79.2, 78.2), "l8_gain": (76.7, 97.1), "l8_consistent": (79.3, 80.8),
         "l16_gain": (75.6, 98.3), "l16_consistent": (79.3, 81.2)}
FOLD_SHARES = {1.0: (0.700, 0.067), 2.0: (0.433, 0.011), 4.0: (0.056, 0.0), 8.0: (0.056, 0.0), 16.0: (0.0, 0.0)}


def table():
    """One row per PENS configuration: mean balanced accuracy and reduction, certified shares, selections.

    For each of the ten configurations (eps_scale in {1, 2, 4, 8, 16} x completion in {consistent, gain}) it reads
    - the test balanced accuracy and reduction on all 90 outer folds from results/real_raw.csv (rows with method
      "pens"), averaged over folds within a dataset and then over the 18 datasets, as in select_real.py;
    - cert_h0 and cert_h1, the number of finite bars of Dgm(K_c) longer than 4 eps_c per class row (dataset x fold x
      class) of results/certificate_real.csv with a nonempty kept set, reduced to the share of class rows with at
      least one certified bar in each degree;
    - how often the trade-off rule selected the configuration, from results/real.csv.
    The per-configuration means and the per-fold certified shares by eps_scale (as in certificate_real.txt) are
    asserted against MEANS and FOLD_SHARES, the values checked on 2026-10-23, and the certified shares are checked
    not to depend on completion (the kept sets do not).

    Returns:
        DataFrame with columns config, eps_scale, completion, bal_acc, reduction, share_cert_h0, share_cert_h1,
        n_class_rows and n_selected_folds.

    Raises:
        AssertionError: If the selections do not sum to 90 folds, a value departs from MEANS or FOLD_SHARES, or the
            certified shares differ between the two completions at some eps_scale.
    """
    raw = pd.read_csv(RESULTS / "real_raw.csv")
    pens = raw[raw.method == "pens"]
    means = pens.groupby(["config", "dataset"], sort=False)[["bal_acc", "reduction"]].mean().groupby("config").mean()
    cert = pd.read_csv(RESULTS / "certificate_real.csv")
    cert = cert[cert.n_kept > 0]
    real = pd.read_csv(RESULTS / "real.csv")
    picks = real[real.method == "pens"].config.value_counts()
    rows = []
    for label, cfg in CONFIGS.items():
        c = cert[cert.config == label]
        rows.append(dict(config=label, eps_scale=cfg["eps_scale"], completion=cfg["completion"],
                         bal_acc=means.loc[label, "bal_acc"], reduction=means.loc[label, "reduction"],
                         share_cert_h0=(c.cert_h0 >= 1).mean(), share_cert_h1=(c.cert_h1 >= 1).mean(),
                         n_class_rows=len(c), n_selected_folds=int(picks.get(label, 0))))
    df = pd.DataFrame(rows)
    assert df.n_selected_folds.sum() == 90, df.n_selected_folds.sum()
    for label, (ba, red) in MEANS.items():
        r = df[df.config == label].iloc[0]
        assert abs(100 * r.bal_acc - ba) <= 0.05 + 1e-9 and abs(100 * r.reduction - red) <= 0.05 + 1e-9, (label, r)
    folds = cert.groupby(["dataset", "fold", "config"]).agg(
        h0=("cert_h0", "sum"), h1=("cert_h1", "sum"), eps_scale=("eps_scale", "first"))
    shares = folds.groupby("eps_scale").agg(h0=("h0", lambda v: (v > 0).mean()), h1=("h1", lambda v: (v > 0).mean()))
    for lam, (h0, h1) in FOLD_SHARES.items():
        got = shares.loc[lam]
        assert abs(got.h0 - h0) <= 5e-4 and abs(got.h1 - h1) <= 5e-4, (lam, got.h0, got.h1)
    # The kept sets do not depend on completion, so neither do the certified shares.
    for lam in LAMBDAS:
        pair = df[df.eps_scale == lam]
        assert pair.share_cert_h0.nunique() == 1 and pair.share_cert_h1.nunique() == 1, lam
    return df


def _axis(ax, panel, title, ylabel, picks_by_lambda):
    """Style one panel: log2 x-axis over LAMBDAS with the selection counts as tick labels, titles and spines.

    Args:
        ax: matplotlib Axes.
        panel: Panel letter for the bold "(a)"-style prefix of the title.
        title: Panel title.
        ylabel: Label of the y-axis.
        picks_by_lambda: Dict mapping eps_scale to the number of folds (of 90) in which the trade-off rule
            selected it, shown under the tick.

    Returns:
        None.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{lam:g}\n({picks_by_lambda[lam]})" for lam in LAMBDAS])
    ax.minorticks_off()
    ax.set_xlabel(f"Radius scale {LAMBDA} (in parentheses: folds in which the trade-off rule selected it, of 90)",
                  color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(rf"$\bf{{({panel})}}$ {title}", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(color=INK_2, labelcolor=INK)


def _series(ax, df, column, color, completion):
    """Plot one column of the table (in %) against eps_scale for one completion.

    Args:
        ax: matplotlib Axes.
        df: Table from table().
        column: Column of df to plot (a share in [0, 1]).
        color: Line color.
        completion: "consistent" (solid, filled markers) or "gain" (dashed, open markers), per STYLE.

    Returns:
        None.
    """
    d = df[df.completion == completion].sort_values("eps_scale")
    ax.plot(d.eps_scale, 100 * d[column], STYLE[completion], color=color, lw=1.8, marker="o", ms=4,
            markerfacecolor="white" if completion == "gain" else color, markeredgewidth=1.4)


def figure(df, path):
    """Draw the two-panel trade-off figure and save it as PDF and PNG.

    Panel (a): mean balanced accuracy and reduction over the 18 datasets against eps_scale, one line per
    completion. Panel (b): share of class rows with a certified bar in degrees 0 and 1; the kept sets, and so the
    certified bars, do not depend on completion, so one line per degree.

    Args:
        df: Table from table().
        path: Output path without suffix; .pdf and .png (200 dpi, white background) are written.

    Returns:
        None.
    """
    picks = df.groupby("eps_scale").n_selected_folds.sum().to_dict()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for completion in COMPLETIONS:
        _series(left, df, "bal_acc", BLUE, completion)
        _series(left, df, "reduction", ORANGE, completion)
    # The kept sets, and so the certified bars, do not depend on completion: one line per degree.
    _series(right, df, "share_cert_h0", BLUE, "consistent")
    _series(right, df, "share_cert_h1", ORANGE, "consistent")
    _axis(left, "a", "Balanced accuracy and reduction", "Mean over the 18 datasets (%)", picks)
    _axis(right, "b", "Classes with a certified bar", "Share of class rows (%)", picks)
    left.set_ylim(0, 102)
    right.set_ylim(-2, 62)
    handles = [Line2D([], [], color=BLUE, lw=1.8, label="Balanced accuracy"),
               Line2D([], [], color=ORANGE, lw=1.8, label="Reduction"),
               Line2D([], [], color=INK, lw=1.4, ls=STYLE["consistent"], label="Consistent completion"),
               Line2D([], [], color=INK, lw=1.4, ls=STYLE["gain"], label="Gain completion")]
    left.legend(handles=handles, loc="center right", frameon=False, labelcolor=INK)
    handles = [Line2D([], [], color=BLUE, lw=1.8, label=r"Degree 0 (bar of Dgm($K_c$) longer than $4\varepsilon_c$)"),
               Line2D([], [], color=ORANGE, lw=1.8, label="Degree 1")]
    right.legend(handles=handles, loc="upper right", frameon=False, labelcolor=INK)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    """Real-data arm: accuracy, reduction and certified shares of PENS across eps_scale, from the result files.

    No new fits. Run after select_real.py and certify_real.py: python plot_lambda.py (no arguments). Reads
    results/real_raw.csv, results/certificate_real.csv and results/real.csv (see table).

    Returns:
        None. Writes results/lambda_tradeoff.csv (one row per PENS configuration) and figures/lambda_tradeoff.pdf
        and figures/lambda_tradeoff.png, and prints the table.
    """
    df = table()
    FIGURES.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "lambda_tradeoff.csv", index=False)
    figure(df, FIGURES / "lambda_tradeoff")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

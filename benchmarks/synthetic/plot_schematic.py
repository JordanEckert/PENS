import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

from pens import PENS  # noqa: E402
from pens.dendrogram import node_class_counts, node_purity_at_least, single_linkage  # noqa: E402
from topology import HERE, RESULTS  # noqa: E402

FIGURES = HERE / "figures"
SEED, N, NOISE, N_FLIP, LAMBDA = 1, 240, 0.15, 19, 4.0
# Figure 1 caption.
EXPECTED = dict(accepted=29, by_purity=25, by_size=4, removed=19, removed_flipped=16, K0=109, K1=112, Q0=18, Q1=15,
                eps=0.335, P0=23, P1=18, flipped_kept=3)
INK, INK_2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e5e4e0", "#2a78d6", "#eb6834"
LIGHT = {0: "#cfe0f7", 1: "#fad8cb"}
COLOR = {0: BLUE, 1: ORANGE}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "ps.fonttype": 42})


def dataset():
    """The Figure 1 dataset: scikit-learn two moons with a fixed set of flipped labels.

    make_moons(n_samples=N, noise=NOISE, random_state=SEED), i.e. 120 points per class with Gaussian noise of
    standard deviation 0.15, with N_FLIP = 19 of the 240 labels (8%) flipped at the rows
    default_rng(SEED).choice(N, N_FLIP, replace=False). This construction, not generators.two_moons, reproduces
    every value in the caption; the seed was recovered by searching seeds and constructions for one that matches
    them all.

    Args:
        None (no arguments); N, NOISE, SEED and N_FLIP are module constants.

    Returns:
        Tuple (X, y, flipped): the (240, 2) coordinates, the labels after flipping, and the sorted row indices of
        the flipped labels.
    """
    X, y = make_moons(n_samples=N, noise=NOISE, random_state=SEED)
    y = np.asarray(y).copy()
    flipped = np.sort(np.random.default_rng(SEED).choice(N, size=N_FLIP, replace=False))
    y[flipped] = 1 - y[flipped]
    return X, y, flipped


def accepted_nodes(X, y, model):
    """Recompute the accepted nodes of the pooled single-linkage tree, as pens.dendrogram.accepted_leaf_assignment.

    A node is accepted when it passes the purity test (node_purity_at_least at the model's purity, with the class
    totals as weights) or has at most s_min points; the highest accepted node on each root-to-leaf path claims
    every node below it.

    Args:
        X: (n, d) coordinates.
        y: (n,) labels, two classes.
        model: Fitted PENS; its metric, metric_params, purity, s_min and clusters_ are used.

    Returns:
        Tuple (Z, assign, pure, claim): the SciPy linkage matrix; for every tree node (leaves 0..n-1, then internal
        nodes n..2n-2) the index of its accepted node, or -1 above the accepted nodes; the boolean purity test per
        node; and the dict accepted node -> class that claims it, read off model.clusters_.
    """
    _, y_enc = np.unique(y, return_inverse=True)
    totals = np.bincount(y_enc, minlength=2)
    Z = single_linkage(X, model.metric, model.metric_params)
    counts = node_class_counts(Z, y_enc, 2)
    n, n_nodes = len(X), len(counts)
    pure = node_purity_at_least(counts, model.purity, totals)
    accept = pure | (counts.sum(axis=1) <= model.s_min)
    assign = np.full(n_nodes, -1, dtype=int)
    for node in range(n_nodes - 1, -1, -1):
        if assign[node] == -1 and accept[node]:
            assign[node] = node
        if node >= n and assign[node] != -1:
            assign[int(Z[node - n, 0])] = assign[node]
            assign[int(Z[node - n, 1])] = assign[node]
    claim = {int(assign[kept[0]]): int(c) for c, kept in model.clusters_}
    return Z, assign, pure, claim


def _axis(ax, panel, title):
    """Style a panel: bold panel letter and title at the left, no top or right spine, ink-coloured ticks.

    Args:
        ax: matplotlib Axes.
        panel: Panel letter, e.g. "a".
        title: Title text placed after the letter.

    Returns:
        None.
    """
    ax.set_title(rf"$\bf{{({panel})}}$ {title}", loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(color=INK_2, labelcolor=INK)


def _legend_below(ax, handles):
    """Place a two-column legend without a frame below a panel.

    Args:
        ax: matplotlib Axes.
        handles: Legend handles (Line2D proxies).

    Returns:
        None.
    """
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False,
              labelcolor=INK, columnspacing=1.2, handletextpad=0.5)


def _scatter_axis(ax, panel, title, X):
    """Style a data panel: equal aspect, limits padded by 0.25 around X, labels x_1 and x_2, grid, and _axis.

    Args:
        ax: matplotlib Axes.
        panel: Panel letter.
        title: Panel title.
        X: (n, 2) coordinates that set the limits.

    Returns:
        None.
    """
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlim(X[:, 0].min() - 0.25, X[:, 0].max() + 0.25)
    ax.set_ylim(X[:, 1].min() - 0.25, X[:, 1].max() + 0.25)
    ax.set_xlabel(r"$x_1$", color=INK)
    ax.set_ylabel(r"$x_2$", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _axis(ax, panel, title)


def tree_panel(ax, X, Z, assign, pure, claim, removed):
    """Panel (a): the pooled single-linkage dendrogram with the accepted nodes and the edit.

    The links under every accepted node are coloured by the class that claims it and the links above the accepted
    nodes are grey; every accepted node is marked at its top, filled when accepted by purity and open when accepted
    by size alone; the removed points are marked with a cross under their leaves.

    Args:
        ax: matplotlib Axes spanning the top row.
        X: (n, 2) coordinates (only n is used).
        Z: SciPy linkage matrix.
        assign: Accepted node of every tree node, -1 above the accepted nodes (from accepted_nodes).
        pure: Boolean purity test per node.
        claim: Dict accepted node -> claiming class.
        removed: Row indices of the points the edit removed.

    Returns:
        None.
    """
    n = len(X)
    order = []
    colors = {node: COLOR[claim[acc]] if acc != -1 else INK_2 for node, acc in enumerate(assign)}
    R = dendrogram(Z, ax=ax, no_labels=True, link_color_func=lambda k: order.append(k) or colors[k])
    for line in ax.collections:
        line.set_linewidth(0.7)
    leaf_x = {leaf: 5.0 + 10.0 * i for i, leaf in enumerate(R["leaves"])}
    node_x = {k: 0.5 * (R["icoord"][i][1] + R["icoord"][i][2]) for i, k in enumerate(order)}
    node_x.update(leaf_x)
    node_h = {k: Z[k - n, 2] if k >= n else 0.0 for k in node_x}
    for acc in claim:
        ax.plot(node_x[acc], node_h[acc], marker="o", ms=4.0, color=COLOR[claim[acc]],
                markerfacecolor=COLOR[claim[acc]] if pure[acc] else "white", markeredgewidth=0.9, zorder=5)
    ax.scatter([leaf_x[i] for i in removed], np.full(len(removed), -0.012), marker="x", s=14, color=INK, lw=0.8,
               zorder=5, clip_on=False)
    ax.set_ylim(-0.03, Z[:, 2].max() * 1.05)
    ax.set_xlim(0, 10.0 * n)
    ax.set_xticks([])
    ax.set_ylabel("Merge height", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    _axis(ax, "a", "Pooled single-linkage tree, accepted nodes and the edit")
    handles = [Line2D([], [], color=COLOR[0], lw=1.2, label="Claimed by class 0"),
               Line2D([], [], color=COLOR[1], lw=1.2, label="Claimed by class 1"),
               Line2D([], [], color=INK_2, lw=1.2, label="Above the accepted nodes"),
               Line2D([], [], ls="", marker="o", color=INK, ms=4, label="Accepted by purity"),
               Line2D([], [], ls="", marker="o", color=INK, markerfacecolor="white", ms=4, label="Accepted by size"),
               Line2D([], [], ls="", marker="x", color=INK, ms=4, label="Removed point")]
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False, labelcolor=INK)


def removed_panel(ax, X, y, removed, flipped):
    """Panel (b): the data with the removed points and the flipped labels.

    Kept points are faint dots and removed points crosses, both coloured by observed label; flipped labels are
    ringed. The title counts the removed points and how many of them carry a flipped label.

    Args:
        ax: matplotlib Axes.
        X: (n, 2) coordinates.
        y: (n,) observed labels.
        removed: Row indices of the removed points.
        flipped: Row indices of the flipped labels.

    Returns:
        None.
    """
    kept = np.setdiff1d(np.arange(len(X)), removed)
    for c in (0, 1):
        idx = kept[y[kept] == c]
        ax.scatter(X[idx, 0], X[idx, 1], s=9, color=COLOR[c], alpha=0.35, lw=0, zorder=2)
        idx = removed[y[removed] == c]
        ax.scatter(X[idx, 0], X[idx, 1], s=30, marker="x", color=COLOR[c], lw=1.1, zorder=4)
    ax.scatter(X[flipped, 0], X[flipped, 1], s=70, facecolors="none", edgecolors=INK, lw=0.7, zorder=5)
    _scatter_axis(ax, "b", f"Removed points ({len(removed)}, of which {len(np.intersect1d(removed, flipped))} "
                  "flipped)", X)
    handles = [Line2D([], [], ls="", marker="o", color=COLOR[0], alpha=0.5, ms=4, label="Kept, label 0"),
               Line2D([], [], ls="", marker="o", color=COLOR[1], alpha=0.5, ms=4, label="Kept, label 1"),
               Line2D([], [], ls="", marker="x", color=INK, ms=5, label="Removed"),
               Line2D([], [], ls="", marker="o", color=INK, markerfacecolor="none", ms=7, label="Flipped label")]
    _legend_below(ax, handles)


def core_panel(ax, X, model):
    """Panel (c): the kept sets K_c with the eps-net cores Q_c and their eps-balls.

    Args:
        ax: matplotlib Axes.
        X: (n, 2) coordinates.
        model: Fitted PENS; kept_, core_ and radii_ are used (the radius is shared by both classes).

    Returns:
        None.
    """
    eps = float(model.radii_[0])
    for c in (0, 1):
        K, Q = np.asarray(model.kept_[c], dtype=int), np.asarray(model.core_[c], dtype=int)
        for q in Q:
            ax.add_patch(Circle(X[q], eps, facecolor=LIGHT[c], edgecolor=COLOR[c], lw=0.5, alpha=0.5, zorder=1))
        ax.scatter(X[K, 0], X[K, 1], s=9, color=COLOR[c], alpha=0.4, lw=0, zorder=2)
        ax.scatter(X[Q, 0], X[Q, 1], s=26, color=COLOR[c], edgecolors=INK, lw=0.5, zorder=4)
    _scatter_axis(ax, "c", rf"Kept sets and $\varepsilon$-net cores, $\varepsilon$ = {eps:.3f}", X)
    handles = [Line2D([], [], ls="", marker="o", color=COLOR[0], alpha=0.5, ms=4, label=r"$K_0$"),
               Line2D([], [], ls="", marker="o", color=COLOR[1], alpha=0.5, ms=4, label=r"$K_1$"),
               Line2D([], [], ls="", marker="o", color=INK_2, markeredgecolor=INK, ms=5,
                      label=r"$Q_c$ with $\varepsilon$-balls")]
    _legend_below(ax, handles)


def prototype_panel(ax, X, y, model):
    """Panel (d): the prototypes P_c, completion points drawn as triangles, with the 1-NN regions they induce.

    The regions are evaluated on a 400 x 400 grid over the padded data range; kept points are drawn faintly, core
    points as circles and completion points (prototypes outside the core) as triangles.

    Args:
        ax: matplotlib Axes.
        X: (n, 2) coordinates.
        y: (n,) observed labels, used to label the prototypes.
        model: Fitted PENS; prototype_indices_, kept_, core_ and prototypes_ are used.

    Returns:
        None.
    """
    P = np.asarray(model.prototype_indices_, dtype=int)
    xs = np.linspace(X[:, 0].min() - 0.25, X[:, 0].max() + 0.25, 400)
    ys = np.linspace(X[:, 1].min() - 0.25, X[:, 1].max() + 0.25, 400)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    region = y[P][cdist(grid, X[P]).argmin(axis=1)].reshape(xx.shape)
    ax.contourf(xx, yy, region, levels=[-0.5, 0.5, 1.5], colors=[LIGHT[0], LIGHT[1]], zorder=0)
    ax.contour(xx, yy, region, levels=[0.5], colors=[INK_2], linewidths=0.7, zorder=1)
    for c in (0, 1):
        K = np.asarray(model.kept_[c], dtype=int)
        ax.scatter(X[K, 0], X[K, 1], s=6, color=COLOR[c], alpha=0.3, lw=0, zorder=2)
        core = np.asarray(model.core_[c], dtype=int)
        comp = np.setdiff1d(np.asarray(model.prototypes_[c], dtype=int), core)
        ax.scatter(X[core, 0], X[core, 1], s=26, color=COLOR[c], edgecolors=INK, lw=0.5, zorder=4)
        ax.scatter(X[comp, 0], X[comp, 1], s=34, marker="^", color=COLOR[c], edgecolors=INK, lw=0.5, zorder=4)
    _scatter_axis(ax, "d", f"Prototypes ({len(model.prototypes_[0])} + {len(model.prototypes_[1])}) and their 1-NN "
                  "regions", X)
    handles = [Line2D([], [], ls="", marker="o", color=INK_2, markeredgecolor=INK, ms=5, label="Core point"),
               Line2D([], [], ls="", marker="^", color=INK_2, markeredgecolor=INK, ms=6, label="Completion point")]
    _legend_below(ax, handles)


def main():
    """Draw Figure 1, one run of PENS on two moons stage by stage, and write the values its caption quotes.

    The data are scikit-learn two moons with 19 flipped labels (dataset); PENS runs at eps_scale = 4 with its other
    defaults. Panels: (a) the pooled single-linkage tree with the accepted nodes and the edit (tree_panel); (b) the
    removed points and the flipped labels (removed_panel); (c) the kept sets with the eps-net cores and their
    eps-balls (core_panel); (d) the prototypes with the 1-NN regions they induce (prototype_panel). The counts the
    caption quotes (EXPECTED) are asserted before drawing, and the text lists every flipped point that survived the
    edit with its role (core, completion, or kept but not a prototype).

    Run from this directory: PYTHONDONTWRITEBYTECODE=1 python plot_schematic.py   (no other script is needed first)

    Args:
        None (no command-line arguments are read); the dataset and PENS settings are the module constants.

    Returns:
        None. Writes figures/schematic.pdf, figures/schematic.png (200 dpi) and results/schematic.txt, and prints
        the text.
    """
    X, y, flipped = dataset()
    model = PENS(eps_scale=LAMBDA).fit(X, y)
    Z, assign, pure, claim = accepted_nodes(X, y, model)
    kept = np.concatenate([np.asarray(model.kept_[c], dtype=int) for c in model.classes_])
    removed = np.setdiff1d(np.arange(N), kept)
    flipped_kept = np.intersect1d(flipped, kept)
    values = dict(accepted=len(claim), by_purity=int(sum(pure[a] for a in claim)),
                  by_size=int(sum(not pure[a] for a in claim)), removed=len(removed),
                  removed_flipped=len(np.intersect1d(removed, flipped)), K0=len(model.kept_[0]),
                  K1=len(model.kept_[1]), Q0=len(model.core_[0]), Q1=len(model.core_[1]),
                  eps=round(float(model.radii_[0]), 3), P0=len(model.prototypes_[0]), P1=len(model.prototypes_[1]),
                  flipped_kept=len(flipped_kept))
    assert float(model.radii_[0]) == float(model.radii_[1])
    for key, expected in EXPECTED.items():
        assert values[key] == expected, (key, values[key], expected)

    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.22, wspace=0.3)
    tree_panel(fig.add_subplot(gs[0, :]), X, Z, assign, pure, claim, removed)
    removed_panel(fig.add_subplot(gs[1, 0]), X, y, removed, flipped)
    core_panel(fig.add_subplot(gs[1, 1]), X, model)
    prototype_panel(fig.add_subplot(gs[1, 2]), X, y, model)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "schematic.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "schematic.png", bbox_inches="tight", dpi=200, facecolor="white")

    lines = [f"Figure 1: sklearn.datasets.make_moons(n_samples={N}, noise={NOISE}, random_state={SEED}), {N_FLIP} "
             f"labels flipped at numpy.random.default_rng({SEED}).choice({N}, {N_FLIP}, replace=False); "
             f"PENS(eps_scale={LAMBDA:g}) with its other defaults. Values checked against the caption.", "",
             f"accepted nodes: {values['accepted']} ({values['by_purity']} by purity, {values['by_size']} by size "
             "only)",
             f"removed points: {values['removed']}, of which {values['removed_flipped']} flipped",
             f"|K_0| = {values['K0']}, |K_1| = {values['K1']}",
             f"|Q_0| = {values['Q0']}, |Q_1| = {values['Q1']}",
             f"eps = {float(model.radii_[0]):.4f}",
             f"|P_0| = {values['P0']}, |P_1| = {values['P1']}",
             f"flipped points surviving the edit: {values['flipped_kept']}"]
    for i in flipped_kept:
        c = int(y[i])
        role = "core" if i in model.core_[c] else "completion" if i in model.prototypes_[c] else "kept, not a prototype"
        lines.append(f"  row {i}: label {c} (flipped from {1 - c}), at ({X[i, 0]:.2f}, {X[i, 1]:.2f}), {role}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "schematic.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

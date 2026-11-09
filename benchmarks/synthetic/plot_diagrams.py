import matplotlib

matplotlib.use("Agg")
import gudhi  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from generators import GENERATORS  # noqa: E402
from pens import PENS  # noqa: E402
from topology import HERE, RESULTS, diagrams  # noqa: E402

FIGURES = HERE / "figures"
GEN, SEED, LAMBDA, CLS = "two_circles", 0, 8.0, 0
INK, INK_2, GRID, BLUE, ORANGE, BAND = "#0b0b0b", "#52514e", "#e5e4e0", "#2a78d6", "#eb6834", "#f1efe9"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "ps.fonttype": 42})


def partner(bar, dgm):
    """The bar of dgm nearest to `bar` in the L-infinity metric, and that distance.

    Args:
        bar: (2,) array (birth, death).
        dgm: (n, 2) array of bars, n >= 1.

    Returns:
        Tuple (nearest bar as a (2,) array, its L-infinity distance to `bar` as a float).
    """
    d = np.abs(dgm - bar).max(axis=1)
    j = int(np.argmin(d))
    return dgm[j], float(d[j])


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


def diagram_panel(ax, panel, k, dK, dP, eps, box=None):
    """Draw the degree-k Rips diagrams of the kept set K_c (open markers) and of the prototypes P_c (filled).

    The band of lifetimes at most 4 eps_c above the diagonal is shaded and the threshold death = birth + 4 eps_c is
    dashed: under the bottleneck bound d_B <= 2 eps_c of Corollary 1 a bar of Dgm(K_c) longer than 4 eps_c cannot be
    matched to the diagonal, so it has a partner in Dgm(P_c) within L-infinity distance 2 eps_c (Corollary 2). When
    `box` is given, a dotted square of half-width 2 eps_c is drawn around that bar: its partner must lie inside.

    Args:
        ax: matplotlib Axes.
        panel: Panel letter.
        k: Homology degree, 0 or 1.
        dK: Diagram list of K_c from topology.diagrams.
        dP: Diagram list of P_c.
        eps: eps_c, the class radius.
        box: (birth, death) of the certified bar to box, or None.

    Returns:
        None.
    """
    lim = max(dK[k][:, 1].max(), dP[k][:, 1].max()) * 1.08 if len(dK[k]) else 1.0
    ax.fill_between([0, lim], [0, lim], [4 * eps, lim + 4 * eps], color=BAND, lw=0, zorder=0)
    ax.plot([0, lim], [0, lim], color=INK_2, lw=0.8, zorder=1)
    ax.plot([0, lim], [4 * eps, lim + 4 * eps], color=INK_2, lw=0.8, ls="--", zorder=1)
    ax.scatter(dK[k][:, 0], dK[k][:, 1], s=26, facecolors="none", edgecolors=BLUE, lw=1.0, zorder=3,
               label=r"$\mathrm{Dgm}(K_c)$")
    ax.scatter(dP[k][:, 0], dP[k][:, 1], s=16, color=ORANGE, zorder=4, label=r"$\mathrm{Dgm}(P_c)$")
    if box is not None:
        ax.add_patch(Rectangle((box[0] - 2 * eps, box[1] - 2 * eps), 4 * eps, 4 * eps, fill=False, ls=":",
                               lw=1.0, edgecolor=INK, zorder=5))
    ax.set_xlim(-0.02 * lim, lim)
    ax.set_ylim(-0.02 * lim, lim + 0.02 * lim)
    ax.set_xlabel("Birth", color=INK)
    ax.set_ylabel("Death", color=INK)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _axis(ax, panel, f"Degree {k}")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)


def main():
    """Draw Figure 2, the certificate on one class of two circles, and write the values its caption quotes.

    Two circles, seed 0, PENS at eps_scale = 8 with consistent completion and its other defaults; class 0 is the
    unit circle (K_c = X_c, 400 points). Panel (a) shows the class, its prototypes and their eps-balls; panels (b)
    and (c) show the Rips diagrams of K_c and P_c in degrees 0 and 1 (diagram_panel), with the certified bar of K_c
    (the longest degree-1 bar, lifetime above 4 eps_c) boxed in degree 1. The quoted values (n_proto, eps, the
    certified lifetime and that of its partner, dB0, dB1) are asserted equal, to a relative 1e-9, to the matching
    row of results/certificate.csv.

    Run after run_certificate.py, from this directory: PYTHONDONTWRITEBYTECODE=1 python plot_diagrams.py

    Args:
        None (no command-line arguments are read); the generator, seed, eps_scale and class are the module
            constants GEN, SEED, LAMBDA and CLS.

    Returns:
        None. Reads results/certificate.csv; writes figures/diagrams.pdf, figures/diagrams.png (200 dpi) and
        results/diagrams.txt, and prints the text.
    """
    X, y, _ = GENERATORS[GEN](SEED)
    m = PENS(eps_scale=LAMBDA, completion="consistent").fit(X, y)
    K, P, eps = np.asarray(m.kept_[CLS], dtype=int), np.asarray(m.prototypes_[CLS], dtype=int), float(m.radii_[CLS])
    dK, dP = diagrams(X[K], 1), diagrams(X[P], 1)
    dB = [float(gudhi.bottleneck_distance(dP[k], dK[k])) for k in (0, 1)]
    life_K = dK[1][:, 1] - dK[1][:, 0]
    certified = dK[1][np.argmax(life_K)]
    mate, dist = partner(certified, dP[1])
    values = dict(n_class=len(np.flatnonzero(y == CLS)), n_kept=len(K), n_proto=len(P), eps=eps, two_eps=2 * eps,
                  four_eps=4 * eps, certified_lifetime=float(life_K.max()),
                  n_certified=int(np.sum(life_K > 4 * eps)), partner_lifetime=float(mate[1] - mate[0]),
                  partner_distance=dist, dB0=dB[0], dB1=dB[1])
    cert = pd.read_csv(RESULTS / "certificate.csv", keep_default_na=False, na_values=["nan", ""])
    row = cert[(cert.gen == GEN) & (cert.seed == SEED) & (cert.eps_scale == LAMBDA) & (cert.completion == "consistent")
               & (cert.cls == CLS)].iloc[0]
    checks = {"n_proto": row.n_proto, "eps": row.eps, "certified_lifetime": row.h1_K_1, "partner_lifetime": row.h1_P_1,
              "dB0": row.dB0_PK, "dB1": row.dB1_PK}
    for key, stored in checks.items():
        assert abs(values[key] - stored) <= 1e-9 * max(1.0, abs(stored)), (key, values[key], stored)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.6))
    idx = np.flatnonzero(y == CLS)
    a.scatter(X[idx, 0], X[idx, 1], s=8, color=BLUE, alpha=0.45, lw=0, zorder=2, label=r"$X_c = K_c$")
    for p in P:
        a.add_patch(Circle(X[p], eps, fill=True, facecolor=ORANGE, alpha=0.12, lw=0))
        a.add_patch(Circle(X[p], eps, fill=False, edgecolor=ORANGE, lw=0.6, alpha=0.8))
    a.scatter(X[P, 0], X[P, 1], s=22, color=ORANGE, zorder=3, label=r"$P_c$")
    a.set_aspect("equal")
    a.set_xlabel(r"$x_1$", color=INK)
    a.set_ylabel(r"$x_2$", color=INK)
    _axis(a, "a", rf"Class and prototypes, $\varepsilon_c$ = {eps:.3f}")
    a.legend(loc="upper right", frameon=False, labelcolor=INK)
    diagram_panel(b, "b", 0, dK, dP, eps)
    diagram_panel(c, "c", 1, dK, dP, eps, box=certified)
    fig.tight_layout(w_pad=2.0)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "diagrams.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "diagrams.png", bbox_inches="tight", dpi=200, facecolor="white")

    lines = [f"Figure 2: {GEN}, seed {SEED}, eps_scale {LAMBDA:g}, consistent completion, class {CLS} (the unit "
             "circle). Values checked against results/certificate.csv.", ""]
    lines += [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in values.items()]
    lines.append(f"The certified bar's partner in Dgm(P_c) lies at L-infinity distance {dist:.4f} <= 2 eps_c = "
                 f"{2 * eps:.4f} from it.")
    text = "\n".join(lines) + "\n"
    (RESULTS / "diagrams.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

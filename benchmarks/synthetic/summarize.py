import numpy as np
import pandas as pd

from generators import GENERATORS
from topology import RESULTS

VARIANT_NAMES = {"pens": "PENS (defaults)", "gain": "completion=gain", "no_completion": "completion=None",
                 "counted": "class_weight=None", "no_prune": "prune=False"}


def fmt(df, floats=3):
    """Render a DataFrame as text with a fixed number of decimals.

    Args:
        df: pandas DataFrame.
        floats: Number of decimals.

    Returns:
        The string from df.to_string.
    """
    return df.to_string(float_format=lambda v: f"{v:.{floats}f}")


def certificate(lines):
    """Append the tables of the certificate arm, read from results/certificate.csv.

    Blocks: how often the certificate (Corollary 1) and the end-to-end bound (Theorem 3) hold; the median
    d_H(P, K) / eps and median max_k d_B(P, K) / (2 eps) by eps_scale (rows with eps > 0); per generator, class and
    degree with beta_k > 0 and per completion, the largest eps_scale at which the number of degree-k bars of
    Dgm(K_c) longer than 4 eps is at least the true beta_k in all 5 seeds, with the median retention |P_c| / n_c
    there; the share of rows of classes with beta1 = 0 that certify a (spurious) degree-1 bar, by eps_scale; for
    the degree-2 classes under consistent completion, the seeds whose kept set has a degree-2 bar longer than 4 eps,
    the median lifetime of its longest degree-2 bar, median 4 eps and median retention; per generator, the median
    bottleneck distances to the whole class X_c of the kept set (K), the prototypes (P), random subsets (R) and the
    farthest-point net (F) of the size of P (consistent completion, at least two prototypes); and the share of rows
    in which P is at least as close to X_c as R and as F in degree 1.

    Args:
        lines: List of text lines, extended in place.

    Returns:
        None.
    """
    df = pd.read_csv(RESULTS / "certificate.csv")
    rated = df[df.holds.notna()]
    e2e = df[df.e2e_holds.notna()]
    lines += ["== Certificate arm ==",
              f"Certificate holds in {int(rated.holds.sum())} of {len(rated)} rows (classes with at least two kept "
              f"points); the end-to-end bound holds in {int(e2e.e2e_holds.sum())} of {len(e2e)} rows "
              f"(at least two prototypes and two kept points).", ""]
    r = rated[rated.eps > 0]
    lines += ["Median d_H(P, K) / eps and median max_k d_B(P, K) / (2 eps) by eps_scale:",
              fmt(pd.DataFrame({"dH/eps": (r.dH_PK / r.eps).groupby(r.eps_scale).median(),
                                "dB/bound": r.tight.groupby(r.eps_scale).median()})), ""]

    # Largest eps_scale at which every seed certifies at least beta_k bars of degree k.
    out = []
    for (gen, cls), g in df[(df.beta1_true > 0) | (df.beta2_true > 0)].groupby(["gen", "cls"], sort=False):
        for k in (1, 2):
            beta = int(g[f"beta{k}_true"].iloc[0])
            if beta == 0:
                continue
            rec = {"gen": gen, "cls": cls, "degree": k, "beta": beta}
            for comp in ("consistent", "gain"):
                h = g[g.completion == comp]
                ok = h.groupby("eps_scale").apply(lambda s: bool((s[f"cert_h{k}"] >= beta).all()), include_groups=False)
                best = ok[ok].index.max() if ok.any() else np.nan
                rec[f"eps_{comp}"] = best
                rec[f"retention_{comp}"] = (np.nan if np.isnan(best) else
                                            (h[h.eps_scale == best].n_proto / h[h.eps_scale == best].n_c).median())
            out.append(rec)
    lines += ["Largest eps_scale at which the number of degree-k bars of Dgm(K_c) longer than 4 eps is at least the "
              "true beta_k in all 5 seeds, and the median retention |P_c| / n_c there (none: no eps_scale qualifies):",
              pd.DataFrame(out).set_index(["gen", "cls", "degree"]).to_string(
                  float_format=lambda v: f"{v:.2f}", na_rep="none"), ""]
    z = df[(df.beta1_true == 0) & df.cert_h1.notna()]
    share = (z.cert_h1 >= 1).groupby(z.eps_scale).mean()
    lines += ["Share of rows of classes with beta1 = 0 that certify at least one degree-1 bar, by eps_scale: "
              + ", ".join(f"{e:g}: {v:.2f}" for e, v in share.items()),
              "(Degree 2 is computed only for classes with beta2 = 1, so there is no such count in degree 2.)", ""]
    s = df[(df.beta2_true > 0) & (df.completion == "consistent")]
    t = s.groupby(["gen", "cls", "eps_scale"]).agg(
        certified_seeds=("cert_h2", lambda x: int((x >= 1).sum())), void_lifetime_K=("h2_K_1", "median"),
        four_eps=("eps", lambda x: 4 * x.median()), retention=("n_proto", "median"), n_c=("n_c", "first"))
    t["retention"] = t.retention / t.n_c
    lines += ["Degree 2, consistent completion: seeds (of 5) whose kept set has a degree-2 bar longer than 4 eps, "
              "median lifetime of the longest degree-2 bar of K_c, median 4 eps and median retention |P_c| / n_c:",
              fmt(t.drop(columns="n_c"), 3), ""]

    # Distances to the whole class (consistent completion, at least two prototypes), medians over seeds and eps_scale.
    c = df[(df.completion == "consistent") & (df.n_proto >= 2)]
    rows = []
    for gen in GENERATORS:
        g = c[c.gen == gen]
        rec = {"gen": gen, "kept share": (g.n_kept / g.n_c).median()}
        for tag in ("KX", "PX", "RX", "FX"):
            rec[f"dB1_{tag}"] = g[f"dB1_{tag}"].median()
        s = g[g.maxdim == 2]
        for tag in ("KX", "PX", "RX", "FX"):
            rec[f"dB2_{tag}"] = s[f"dB2_{tag}"].median() if len(s) else np.nan
        rows.append(rec)
    lines += ["Distance to the whole class X_c (consistent completion, rows with at least two prototypes; medians over "
              "seeds, classes and eps_scale). K = kept set, P = prototypes, R = random subsets and F = farthest-point "
              "net of X_c, both of the size of P; degree 2 only for classes with beta2 > 0:",
              fmt(pd.DataFrame(rows).set_index("gen")), ""]
    both = c[c.dB1_RX.notna()]
    lines += [f"P at least as close as random in degree 1: {(both.dB1_PX <= both.dB1_RX + 1e-12).mean():.2f} of rows; "
              f"as close as the farthest-point net: {(both.dB1_PX <= both.dB1_FX + 1e-12).mean():.2f}.", ""]


def ablation(lines):
    """Append the tables of the noise and imbalance ablations, read from results/ablation.csv.

    For each sweep (columns: label-noise rate, or imbalance ratio with the noise-free condition reused as ratio 1),
    tables indexed by eps_scale x variant with means over generators, seeds and classes: clean-test balanced
    accuracy, retention, the share of mislabeled (noise sweep only) and of correctly labeled points removed by the
    edit, for the imbalance sweep the share of the minority class removed, d_B in degree 1 from the prototypes to
    the clean class for the loop classes and in degree 2 for the sphere classes, and the share of loop and sphere
    classes whose loop or void is certified (cert_h1 >= 1, cert_h2 >= 1); for the noise sweep also the balanced
    accuracy by generator at the PENS defaults.

    Args:
        lines: List of text lines, extended in place.

    Returns:
        None.
    """
    df = pd.read_csv(RESULTS / "ablation.csv")
    lines += ["== Noise and imbalance ablations (eps_scale = 1 and 4) ==",
              f"Certificate holds in {int(df.holds.sum())} of {int(df.holds.notna().sum())} rows.", ""]
    per_run = df.drop_duplicates(["gen", "seed", "sweep", "noise", "ir", "eps_scale", "variant"])
    idx = ["eps_scale", "variant"]
    for sweep, key in (("noise", "noise"), ("imbalance", "ir")):
        base = df[(df.sweep == "noise") & (df.noise == 0)].assign(sweep=sweep) if sweep == "imbalance" else None
        d = pd.concat([base, df[df.sweep == sweep]]) if base is not None else df[df.sweep == sweep]
        r = per_run[per_run.sweep == sweep]
        if base is not None:
            r = pd.concat([per_run[(per_run.sweep == "noise") & (per_run.noise == 0)], r])
        tables = [("Clean-test balanced accuracy (%)", 100 * r.pivot_table("test_bal_acc", idx, key)),
                  ("Retention (%)", 100 * r.pivot_table("retention", idx, key))]
        if sweep == "noise":
            tables.append(("Share of mislabeled points removed by the edit (%)",
                           100 * d.pivot_table("removed_mislabeled", idx, key)))
        tables.append(("Share of correctly labeled points removed by the edit (%)",
                       100 * d.pivot_table("removed_clean", idx, key)))
        if sweep == "imbalance":
            m = d[d.cls == 1]
            tables.append(("Minority class: share of its points removed by the edit (%)",
                           100 * m.pivot_table("removed_clean", idx, key)))
        loop = d[(d.beta1_true > 0)]
        tables.append(("d_B in degree 1 from the prototypes to the clean class (classes with beta1 = 1)",
                       loop.pivot_table("dB1_PC", idx, key)))
        sph = d[d.beta2_true > 0]
        tables.append(("d_B in degree 2 from the prototypes to the clean sphere",
                       sph.pivot_table("dB2_PC", idx, key)))
        rec1 = loop.assign(ok=loop.cert_h1 >= loop.beta1_true)
        rec2 = sph.assign(ok=sph.cert_h2 >= sph.beta2_true)
        tables.append(("Share of loop classes with the loop certified (cert_h1 >= 1) (%)",
                       100 * rec1.pivot_table("ok", idx, key)))
        tables.append(("Share of sphere classes with the void certified (cert_h2 >= 1) (%)",
                       100 * rec2.pivot_table("ok", idx, key)))
        lines.append(f"-- {sweep} sweep (columns: {'label-noise rate' if sweep == 'noise' else 'imbalance ratio'}; "
                     "means over generators, seeds and classes) --")
        order = pd.MultiIndex.from_product([sorted(df.eps_scale.unique()), list(VARIANT_NAMES)])
        for title, t in tables:
            t = t.reindex(order).rename(index=VARIANT_NAMES, level=1)
            lines += [title, fmt(t, 1 if "%" in title else 3), ""]
        if sweep == "noise":
            lines += ["Clean-test balanced accuracy (%) by generator, PENS defaults:",
                      fmt(100 * r[r.variant == "pens"].pivot_table("test_bal_acc", ["eps_scale", "gen"], key), 1), ""]


def main():
    """Write results/summary.txt, the summary tables of the synthetic arm.

    Run after run_certificate.py and run_ablation.py, from this directory: PYTHONDONTWRITEBYTECODE=1 python summarize.py
    Reads results/certificate.csv and results/ablation.csv; no fits are run, so it takes seconds.

    Args:
        None (no command-line arguments are read).

    Returns:
        None. Writes results/summary.txt and prints the text.
    """
    lines = []
    certificate(lines)
    ablation(lines)
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

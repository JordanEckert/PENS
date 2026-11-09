import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from generators import GENERATORS
from pens import PENS, edit_cost, verify_certificate
from topology import RESULTS, DiagramCache, bott, fps, maxdim, n_longer, top_lifetimes

SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [(lam, completion) for completion in (None, "consistent", "gain") for lam in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
R_RANDOM = 5
NAN = float("nan")
DEG = (0, 1, 2)
COLS = (["gen", "seed", "eps_scale", "completion", "cls", "beta1_true", "beta2_true", "dim", "maxdim", "n_c",
         "n_kept", "n_core", "n_proto", "guarded", "eps", "bound", "dH_PK"] + [f"dB{k}_PK" for k in DEG]
        + ["holds", "tight"] + [f"h{k}_{s}_{i}" for s in "XKP" for k in (1, 2) for i in (1, 2)]
        + ["cert_h1", "cert_h2", "dH_KX"] + [f"dB{k}_KX" for k in DEG] + ["e2e_bound"]
        + [f"dB{k}_PX" for k in DEG] + ["e2e_holds"] + [f"dB{k}_RX" for k in DEG] + [f"dB{k}_FX" for k in DEG])


def lifetimes(row, tag, dgms):
    """Store the two longest lifetimes of every positive degree of a diagram list in a row.

    Args:
        row: Row dict, updated in place.
        tag: "X", "K" or "P", the point set the diagrams belong to.
        dgms: Diagram list [H0, ..., H_dim] from topology.diagrams.

    Returns:
        None; sets row["h{k}_{tag}_1"] and row["h{k}_{tag}_2"] for k = 1..dim.
    """
    for k in range(1, len(dgms)):
        row[f"h{k}_{tag}_1"], row[f"h{k}_{tag}_2"] = top_lifetimes(dgms[k])


def task(args):
    """Fit PENS at every configuration on one generator and seed and record the certificate and topology columns.

    For each (eps_scale, completion) of CONFIGS (eps_scale in {0.5, 1, 2, 4, 8, 16}, completion None, consistent or
    gain, other settings at their defaults) and each class c: the certificate against the kept set K_c (Corollary 1:
    dH_PK = d_H(P_c, K_c), dB*_PK and holds from pens.verify_certificate, extended to degree 2 for classes with
    beta2 > 0; tight = max_k dB{k}_PK / (2 eps), bound = 2 eps); the edit cost dH_KX = d_H(K_c, X_c) and the
    end-to-end bound e2e_bound = 2 (eps + dH_KX) against the whole class X_c (Theorem 3), with e2e_holds testing
    max_k dB{k}_PX <= e2e_bound; the number of bars of Dgm(K_c) longer than 4 eps (cert_h1, cert_h2; by Corollary 2
    these must survive in the prototypes); the two longest lifetimes per positive degree of X_c, K_c and P_c
    (h{k}_{X,K,P}_{1,2}); and the bottleneck distances to X_c of the prototypes (dB*_PX), of R_RANDOM random subsets
    of X_c of the same size (dB*_RX, the mean; rng = default_rng([seed, c, config index])) and of a farthest-point
    net of X_c of the same size (dB*_FX, topology.fps), both 0 when P_c is the whole class. Diagrams go up to
    degree 2 for classes with beta2 > 0 and degree 1 otherwise; guarded = 1 marks a class the edit emptied (a guard
    class), and the certificate columns stay NaN for classes with fewer than two kept points or prototypes.

    Args:
        args: Tuple (generator name, seed).

    Returns:
        List of row dicts with the keys COLS, one per (configuration, class).
    """
    name, seed = args
    X, y, truth = GENERATORS[name](seed)
    dg = DiagramCache(X)
    idx_c = {c: np.flatnonzero(y == c) for c in (0, 1)}
    rows = []
    for ci, (lam, completion) in enumerate(CONFIGS):
        m = PENS(eps_scale=lam, completion=completion).fit(X, y)
        cert = verify_certificate(X, m.kept_, m.prototypes_, m.radii_, maxdim=1)
        cost = edit_cost(X, y, m.kept_)
        for c in (0, 1):
            dim = maxdim(truth[c])
            K, P, eps = np.asarray(m.kept_[c]), np.asarray(m.prototypes_[c]), float(m.radii_[c])
            dX = dg(idx_c[c], dim)
            row = dict.fromkeys(COLS, NAN)
            row.update(gen=name, seed=seed, eps_scale=lam, completion=str(completion), cls=c,
                       beta1_true=truth[c][1], beta2_true=truth[c][2], dim=X.shape[1], maxdim=dim,
                       n_c=len(idx_c[c]), n_kept=len(K), n_core=len(m.core_[c]), n_proto=len(P),
                       guarded=int(len(K) == 0), eps=eps, bound=2 * eps)
            lifetimes(row, "X", dX)
            dP = dg(P, dim) if len(P) >= 2 else None
            if c in cert:
                r = cert[c]
                row.update(dH_PK=r["hausdorff"], dB0_PK=r["bottleneck"][0], dB1_PK=r["bottleneck"][1],
                           holds=bool(r["holds"]))
                if dim == 2:
                    row["dB2_PK"] = bott(dP[2], dg(K, 2)[2]) if dP is not None else bott(np.empty((0, 2)), dg(K, 2)[2])
                    row["holds"] = row["holds"] and row["dB2_PK"] <= 2 * eps + 1e-9
                if eps > 0:
                    row["tight"] = max(row[f"dB{k}_PK"] for k in range(dim + 1)) / (2 * eps)
            if len(K) >= 2:
                dK = dg(K, dim)
                lifetimes(row, "K", dK)
                row["cert_h1"] = n_longer(dK[1], 4 * eps)
                if dim == 2:
                    row["cert_h2"] = n_longer(dK[2], 4 * eps)
                row["dH_KX"] = cost[c]
                for k in range(dim + 1):
                    row[f"dB{k}_KX"] = bott(dK[k], dX[k])
                row["e2e_bound"] = 2 * (eps + cost[c])
            if dP is not None:
                lifetimes(row, "P", dP)
                for k in range(dim + 1):
                    row[f"dB{k}_PX"] = bott(dP[k], dX[k])
                if len(K) >= 2:
                    row["e2e_holds"] = max(row[f"dB{k}_PX"] for k in range(dim + 1)) <= row["e2e_bound"] + 1e-9
                n = len(P)
                if n < len(idx_c[c]):
                    rng = np.random.default_rng([seed, c, ci])
                    draws = [dg(rng.choice(idx_c[c], size=n, replace=False), dim) for _ in range(R_RANDOM)]
                    dF = dg(idx_c[c][fps(X[idx_c[c]], n)], dim)
                    for k in range(dim + 1):
                        row[f"dB{k}_RX"] = float(np.mean([bott(d[k], dX[k]) for d in draws]))
                        row[f"dB{k}_FX"] = bott(dF[k], dX[k])
                else:
                    for k in range(dim + 1):
                        row[f"dB{k}_RX"] = row[f"dB{k}_FX"] = 0.0
            rows.append(row)
    return rows


def sanity():
    """Print the longest H1 and H2 bars of every class of every generator at seed 0.

    A check that the clean shapes carry the Betti numbers the generators claim: a class should show as many long
    bars in degree k as its beta_k, well separated from the rest.

    Args:
        None (no arguments).

    Returns:
        None; prints one line per generator and class with its Betti numbers and the three longest lifetimes per
        positive degree.
    """
    for name, gen in GENERATORS.items():
        X, y, truth = gen(0)
        dg = DiagramCache(X)
        for c in (0, 1):
            d = dg(np.flatnonzero(y == c), maxdim(truth[c]))
            parts = [f"H{k} " + " ".join(f"{v:.3f}" for v in top_lifetimes(d[k], 3)) for k in range(1, len(d))]
            print(f"{name:18s} cls {c} betti {truth[c]}  longest bars: {'; '.join(parts)}", flush=True)


def main():
    """Run the certificate arm on every generator and seed (0-4) in parallel, or the sanity check.

    Run from this directory:
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_certificate.py
        python run_certificate.py sanity
    NPROC (default 4) sets the number of worker processes, one task per generator and seed and one thread each;
    generators with degree-2 classes are dispatched first because they take longest. No other script is required
    first; summarize.py, plot_diagrams.py and run_matched.py read the output.

    Args:
        None, or the single word "sanity" in sys.argv[1:] to run sanity() instead of the sweep.

    Returns:
        None. Writes results/certificate.csv with the columns COLS, rows sorted by generator (GENERATORS order),
        seed, configuration (CONFIGS order) and class; prints progress and the total time.
    """
    if sys.argv[1:] == ["sanity"]:
        return sanity()
    RESULTS.mkdir(exist_ok=True)
    # Generators with degree-2 classes first: they take longest.
    names = sorted(GENERATORS, key=lambda g: -max(t[2] for t in GENERATORS[g](0)[2].values()))
    tasks = [(n, s) for n in names for s in SEEDS]
    t0 = time.time()
    done = []
    with Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(task, tasks):
            done.extend(rows)
            print(f"{rows[0]['gen']} seed {rows[0]['seed']} done (elapsed {time.time() - t0:.0f}s)", flush=True)
    order = list(GENERATORS)
    done.sort(key=lambda r: (order.index(r["gen"]), r["seed"],
                             CONFIGS.index((r["eps_scale"], None if r["completion"] == "None" else r["completion"])),
                             r["cls"]))
    with open(RESULTS / "certificate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(done)
    print(f"TOTAL {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

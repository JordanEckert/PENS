import csv
import os
import time
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from generators import GENERATORS
from pens import PENS, edit_cost, verify_certificate
from topology import RESULTS, DiagramCache, bott, maxdim, n_longer

GENS = ["sphere_blob", "two_circles", "circle_vs_disk", "two_moons"]
SEEDS = [0, 1, 2, 3, 4]
EPS_SCALES = [1.0, 4.0]  # 1 certifies the voids of the clean spheres (see run_certificate.py); 4 is the default
CONDITIONS = ([("noise", rho, 1) for rho in (0.0, 0.05, 0.1, 0.2, 0.3)]
              + [("imbalance", 0.0, ir) for ir in (2, 5, 10, 20)])
VARIANTS = [("pens", {}), ("gain", {"completion": "gain"}), ("no_completion", {"completion": None}),
            ("counted", {"class_weight": None}), ("no_prune", {"prune": False})]
NAN = float("nan")
DEG = (0, 1, 2)
COLS = (["gen", "seed", "sweep", "noise", "ir", "eps_scale", "variant", "cls", "beta1_true", "beta2_true", "maxdim", "n_c",
         "n_clean", "n_mislabeled", "n_kept", "n_proto", "eps", "holds", "dH_KX", "e2e_bound", "removed_mislabeled",
         "removed_clean"] + [f"dB{k}_KC" for k in DEG] + [f"dB{k}_PC" for k in DEG]
        + ["cert_h1", "cert_h2", "retention", "test_bal_acc", "fit_time_s"])


def task(args):
    """Run the label-noise and class-imbalance sweeps on one generator and seed, every eps_scale and variant.

    The generator is drawn at `seed` (training) and at seed + 1000 (a fresh clean test draw). For each condition
    of CONDITIONS: in the noise sweep a fraction rho of the training labels, chosen uniformly without replacement,
    is flipped to the other class; in the imbalance sweep class 1 is subsampled to 1/IR of its size (at least 2
    points); the condition's rng is default_rng([seed, round(100 rho), IR]). PENS is then fitted at every eps_scale
    of EPS_SCALES (1 and 4) with its defaults and with one switch changed (VARIANTS: completion="gain",
    completion=None, class_weight=None, prune=False). One row per class c of the observed labels records: what the
    edit removes, removed_mislabeled being the share of the points labelled c whose true class is not c that the
    edit removed and removed_clean the share of the correctly labelled ones; the certificate against the kept set
    (holds from pens.verify_certificate, extended to degree 2 for classes with beta2 > 0; dH_KX = d_H(K_c, X_c) and
    e2e_bound = 2 (eps + dH_KX), Theorem 3); the bottleneck distances of the kept set (dB*_KC) and of the prototypes
    (dB*_PC) to the clean class, the points whose true class is c; the number of bars of Dgm(K_c) longer than 4 eps
    (cert_h1, cert_h2: the certified bars, Corollary 2) next to the true Betti numbers; retention = |S| / n for the
    full selection S; test_bal_acc, the balanced accuracy of 1-NN on S on the clean test draw; and the fit time.
    Diagrams go up to degree 2 for classes with beta2 > 0 and degree 1 otherwise.

    Args:
        args: Tuple (generator name, seed).

    Returns:
        List of row dicts with the keys COLS, one per (condition, eps_scale, variant, class).
    """
    name, seed = args
    X0, y0, truth = GENERATORS[name](seed)
    Xt, yt, _ = GENERATORS[name](seed + 1000)
    caches = {}
    rows = []
    for sweep, rho, ir in CONDITIONS:
        rng = np.random.default_rng([seed, int(round(100 * rho)), ir])
        keep = np.arange(len(y0))
        if ir > 1:
            minority = np.flatnonzero(y0 == 1)
            sub = rng.choice(minority, size=max(2, int(round(len(minority) / ir))), replace=False)
            keep = np.sort(np.concatenate([np.flatnonzero(y0 == 0), sub]))
        X, y_true = X0[keep], y0[keep]
        y = y_true.copy()
        if rho > 0:
            flip = rng.choice(len(y), size=int(round(rho * len(y))), replace=False)
            y[flip] = 1 - y[flip]
        dg = caches.setdefault(ir, DiagramCache(X))  # the noise conditions share X
        for lam, (variant, cfg) in [(lam, v) for lam in EPS_SCALES for v in VARIANTS]:
            t0 = time.perf_counter()
            m = PENS(eps_scale=lam, **cfg).fit(X, y)
            fit_time = time.perf_counter() - t0
            S = m.prototype_indices_
            pred = KNeighborsClassifier(n_neighbors=1).fit(X[S], y[S]).predict(Xt)
            test_bal_acc = float(balanced_accuracy_score(yt, pred))
            cert = verify_certificate(X, m.kept_, m.prototypes_, m.radii_, maxdim=1)
            cost = edit_cost(X, y, m.kept_)
            for c in (0, 1):
                dim = maxdim(truth[c])
                obs, clean = np.flatnonzero(y == c), np.flatnonzero(y_true == c)
                K, P, eps = np.asarray(m.kept_[c]), np.asarray(m.prototypes_[c]), float(m.radii_[c])
                in_K = np.isin(obs, K)
                wrong = y_true[obs] != c
                row = dict.fromkeys(COLS, NAN)
                row.update(gen=name, seed=seed, sweep=sweep, noise=rho, ir=ir, eps_scale=lam, variant=variant, cls=c,
                           beta1_true=truth[c][1], beta2_true=truth[c][2], maxdim=dim, n_c=len(obs),
                           n_clean=len(clean), n_mislabeled=int(wrong.sum()), n_kept=len(K), n_proto=len(P), eps=eps,
                           retention=len(S) / len(X), test_bal_acc=test_bal_acc, fit_time_s=fit_time)
                if wrong.any():
                    row["removed_mislabeled"] = float(np.mean(~in_K[wrong]))
                if (~wrong).any():
                    row["removed_clean"] = float(np.mean(~in_K[~wrong]))
                dC = dg(clean, dim)
                if c in cert:
                    row["holds"] = bool(cert[c]["holds"])
                    if dim == 2:
                        dB2 = bott(dg(P, 2)[2] if len(P) >= 2 else np.empty((0, 2)), dg(K, 2)[2])
                        row["holds"] = row["holds"] and dB2 <= 2 * eps + 1e-9
                    row["dH_KX"] = cost[c]
                    row["e2e_bound"] = 2 * (eps + cost[c])
                    dK = dg(K, dim)
                    row["cert_h1"] = n_longer(dK[1], 4 * eps)
                    if dim == 2:
                        row["cert_h2"] = n_longer(dK[2], 4 * eps)
                    for k in range(dim + 1):
                        row[f"dB{k}_KC"] = bott(dK[k], dC[k])
                if len(P) >= 2:
                    dP = dg(P, dim)
                    for k in range(dim + 1):
                        row[f"dB{k}_PC"] = bott(dP[k], dC[k])
                rows.append(row)
    return rows


def main():
    """Run the ablation sweeps on GENS x SEEDS (seeds 0-4) in parallel and write results/ablation.csv.

    Run from this directory:
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python run_ablation.py
    NPROC (default 4) sets the number of worker processes, one task per generator and seed and one thread each.
    No other script is required first; summarize.py reads the output.

    Args:
        None (no command-line arguments are read).

    Returns:
        None. Writes results/ablation.csv with the columns COLS, rows sorted by generator (GENS order), seed,
        condition (CONDITIONS order), eps_scale, variant (VARIANTS order) and class; prints progress and the total
        time.
    """
    RESULTS.mkdir(exist_ok=True)
    tasks = [(g, s) for g in GENS for s in SEEDS]
    t0 = time.time()
    done = []
    with Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(task, tasks):
            done.extend(rows)
            print(f"{rows[0]['gen']} seed {rows[0]['seed']} done (elapsed {time.time() - t0:.0f}s)", flush=True)
    cond = [(s, r, i) for s, r, i in CONDITIONS]
    var = [v for v, _ in VARIANTS]
    done.sort(key=lambda r: (GENS.index(r["gen"]), r["seed"], cond.index((r["sweep"], r["noise"], r["ir"])),
                             r["eps_scale"], var.index(r["variant"]), r["cls"]))
    with open(RESULTS / "ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(done)
    print(f"TOTAL {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

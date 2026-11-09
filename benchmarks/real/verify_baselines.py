import contextlib
import io
import subprocess
import sys
import tempfile
import types
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, squareform

import baselines as B
from common import DATASETS, GRIDS, HERE, RESULTS, load, minmax, outer_folds

CACHE = HERE / ".cache"
KEEL = ("keel", "https://github.com/SCI2SUGR/KEEL", "5f404a9470f17528445baabe3a5d9f5202adee7a")
WAASHK = ("waashk", "https://github.com/waashk/instanceselection", "a857d9c88a9d92e55083f8607932502cb989d7bf")
DPR = ("dpr1005", "https://github.com/dpr1005/Semisupervised-learning-and-instance-selection-methods",
       "ef35fda610df37f2cc5f0ad2de912fab7b1090e9")
PROTOCLASS = ("protoclass", "https://github.com/cran/protoclass", "59cd93751bf0f533fd0c7759c6d1c75435388ea3")
KEEL_ALGOS = ["ENN", "CNN", "ICF", "IB3", "HMNEI"]
# The pure-Python references are quadratic in Python loops; by default they run on the datasets below 2000 rows.
SMALL = ["bupa", "ecoli", "ionosphere", "new_thyroid", "sonar", "vehicle", "vowel", "wdbc", "wine_quality_red", "yeast"]


def fetch(repo, paths):
    """Check out `paths` of a repository at a pinned commit into .cache/<name> (once; a marker file skips reruns).

    Args:
        repo: Tuple (name, git URL, commit hash), one of KEEL, WAASHK, DPR, PROTOCLASS.
        paths: Paths inside the repository to check out.

    Returns:
        The checkout directory .cache/<name> as a Path.
    """
    name, url, commit = repo
    dest = CACHE / name
    if not (dest / ".fetched").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git = ["git", "-C", str(dest)]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["fetch", "-q", "--depth", "1", "--filter=blob:none", url, commit], check=True)
        subprocess.run(git + ["checkout", "-q", "FETCH_HEAD", "--", *paths], check=True)
        (dest / ".fetched").touch()
    return dest


def keel_build():
    """Fetch the KEEL sources of the five instance-selection algorithms and compile them with javac (once).

    Returns:
        The directory of the compiled classes, .cache/keel/build, as a Path.
    """
    dirs = ["src/org/core", "src/keel/Dataset", "src/keel/Algorithms/Preprocess/Basic"] + \
           [f"src/keel/Algorithms/Preprocess/Instance_Selection/{a}" for a in KEEL_ALGOS]
    src = fetch(KEEL, dirs)
    build = src / "build"
    if not (build / ".built").exists():
        files = [str(p) for d in dirs for p in (src / d).rglob("*.java")]
        subprocess.run(["javac", "-nowarn", "-encoding", "ISO-8859-1", "-d", str(build), *files], check=True,
                       stderr=subprocess.DEVNULL)
        (build / ".built").touch()
    return build


def folds(name):
    """Yield the min-max scaled training part of every outer fold of a dataset, as in run_real.py.

    Args:
        name: Dataset name from common.DATASETS.

    Yields:
        Tuples (fold index, scaled training features, training labels encoded as 0..C-1).
    """
    X, y = load(name)
    y = np.unique(y, return_inverse=True)[1]
    for fold, (tr, te) in enumerate(outer_folds(X, y)):
        yield fold, minmax(X[tr], X[te])[0], y[tr]


def canonical(X, y):
    """An id per training row; exact duplicates (same features and label) share it.

    Args:
        X: (n, d) training features.
        y: (n,) class codes.

    Returns:
        (n,) integer array of row ids.
    """
    return np.unique(np.column_stack([X, y]), axis=0, return_inverse=True)[1].ravel()


def as_multiset(X, y, idx):
    """Multiset of canonical row ids of a selection, so that exact duplicates are compared by count.

    Args:
        X: (n, d) training features.
        y: (n,) class codes.
        idx: Selected row indices.

    Returns:
        Counter mapping canonical row id to multiplicity.
    """
    return Counter(canonical(X, y)[idx].tolist())


def rows_to_multiset(X, y, rows, labels, tol=1e-4):
    """Match rows written by a reference implementation to the nearest training row of the same class.

    Needed because KEEL writes the selected rows (rounded) rather than their indices.

    Args:
        X: (n, d) training features.
        y: (n,) class codes.
        rows: (k, d) feature rows written by the reference.
        labels: (k,) their class codes.
        tol: Largest allowed max-norm distance to the matched training row.

    Returns:
        Counter mapping canonical row id to multiplicity, comparable with as_multiset.

    Raises:
        AssertionError: If some output row has no training row of its class within tol.
    """
    canon = canonical(X, y)
    ids = []
    for v, c in zip(rows, labels):
        cand = np.flatnonzero(y == c)
        d = np.abs(X[cand] - v).max(axis=1)
        j = int(d.argmin())
        assert d[j] < tol, f"no training row within {tol} of an output row (nearest {d[j]:.3g})"
        ids.append(int(canon[cand[j]]))
    return Counter(ids)


def grid_values(method, key):
    """Values of one parameter over the grid of a method in common.GRIDS.

    Args:
        method: Key of GRIDS.
        key: Parameter name in the configuration dicts.

    Returns:
        List of the parameter's values, in grid order.
    """
    return [cfg[key] for _, cfg in GRIDS[method]]


def compare(a, b):
    """Describe whether two selection multisets agree.

    Args:
        a: Our selection, as a Counter from as_multiset.
        b: The reference selection, as a Counter.

    Returns:
        "identical", or "DIFFERENT (...)" with both sizes and the size of the symmetric difference.
    """
    return "identical" if a == b else f"DIFFERENT (|ours|={sum(a.values())}, |ref|={sum(b.values())}, " \
                                      f"symmetric difference {sum(((a - b) + (b - a)).values())})"


# ---- KEEL ----------------------------------------------------------------------------------------------------------

class KeelRandom:
    """KEEL's Randomize (MT19937 seeded with init_genrand, i.e. numpy's RandomState), with the interface of
    random.Random that the vendored IB3 uses. randint(low, high) never returns high unless low == high, and shuffle
    is KEEL's loop, as in KEEL's IB3."""

    def __init__(self, seed):
        """Seed the generator.

        Args:
            seed: Integer seed (KEEL's default is baselines.KEEL_SEED).
        """
        self.rs = np.random.RandomState(seed)

    def randint(self, low, high):
        """KEEL's Randint: an integer in [low, high), or low when low == high.

        Args:
            low: Lower bound, inclusive.
            high: Upper bound, exclusive.

        Returns:
            The drawn integer.
        """
        return B._keel_randint(self.rs, low, high)

    def shuffle(self, x):
        """Shuffle a list in place with KEEL's loop (swap position i with a random position in [i, len - 1)).

        Args:
            x: List to shuffle.

        Returns:
            None.
        """
        for i in range(len(x)):
            pos = self.randint(i, len(x) - 1)
            x[i], x[pos] = x[pos], x[i]


def write_dat(path, X, y, n_classes):
    """Write a training part as a KEEL .dat file with real attributes in [0, 1] and class labels 0..C-1.

    Args:
        path: Output path.
        X: (n, d) scaled training features.
        y: (n,) class codes.
        n_classes: Number of classes C.

    Returns:
        None.
    """
    names = [f"a{j}" for j in range(X.shape[1])]
    lines = ["@relation verify"] + [f"@attribute {a} real [0.0, 1.0]" for a in names]
    lines += ["@attribute class {" + ",".join(str(c) for c in range(n_classes)) + "}",
              "@inputs " + ",".join(names), "@outputs class", "@data"]
    lines += [",".join(repr(float(v)) for v in row) + f",{c}" for row, c in zip(X, y)]
    path.write_text("\n".join(lines) + "\n")


def run_keel(build, algo, X, y, params):
    """Run one KEEL Preprocess/Instance_Selection algorithm on a training part in a temporary directory.

    Args:
        build: Directory of the compiled KEEL classes (from keel_build).
        algo: KEEL algorithm name (one of KEEL_ALGOS).
        X: (n, d) scaled training features.
        y: (n,) class codes.
        params: List of (KEEL parameter name, value) pairs for the config file; the Euclidean distance is added.

    Returns:
        Tuple (rows, labels): the (k, d) feature rows KEEL wrote to its output file and their (k,) class codes.
    """
    n_classes = int(y.max()) + 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_dat(tmp / "tra.dat", X, y, n_classes)
        conf = [f"algorithm = {algo}", f'inputData = "{tmp / "tra.dat"}" "{tmp / "tra.dat"}"',
                f'outputData = "{tmp / "out.dat"}" "{tmp / "out_tst.dat"}"', ""]
        conf += [f"{k} = {v}" for k, v in params] + ["Distance Function = Euclidean"]
        (tmp / "config.txt").write_text("\n".join(conf) + "\n")
        subprocess.run(["java", "-cp", str(build), f"keel.Algorithms.Preprocess.Instance_Selection.{algo}.Main",
                        str(tmp / "config.txt")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = (tmp / "out.dat").read_text().split("@data", 1)[1]
    rows = [r.split(",") for r in text.split("\n") if r.strip()]
    return np.array([[float(v) for v in r[:-1]] for r in rows]).reshape(-1, X.shape[1]), \
        np.array([int(r[-1].strip()) for r in rows], dtype=int)


def keel_task(args):
    """Compare enn, cnn, icf, hmnei and ib3 with KEEL on one fold, at every configuration of their grids.

    Args:
        args: Tuple (dataset name, fold index, scaled training features, class codes, KEEL build directory).

    Returns:
        List of result lines, one per algorithm and configuration, each ending with compare's verdict.
    """
    name, fold, X, y, build = args

    def ib3(a, d):
        """Our IB3 with KEEL's random generator in place of random.Random, so that it can match KEEL's IB3.

        Args:
            a: Acceptance level.
            d: Drop level.

        Returns:
            Integer array of the selected row indices.
        """
        m = B.IB3(rng=KeelRandom(B.KEEL_SEED))
        m.nAccept, m.nDrop = a, d
        return np.asarray(m.fit(X, y).sample_indices_, dtype=int)

    # Every configuration of the grids in common.GRIDS.
    checks = [("ENN", f"k={k}", [("Number of Neighbors", k)], lambda k=k: B.enn(X, y, k=k))
              for k in grid_values("enn", "k")]
    checks += [("CNN", f"k={k}", [("Random Seed", B.KEEL_SEED), ("Number of Neighbors", k)],
                lambda k=k: B.cnn(X, y, k=k)) for k in grid_values("cnn", "k")]
    checks += [("ICF", f"k={k}", [("Number of Neighbors", k)], lambda k=k: B.icf(X, y, k=k))
               for k in grid_values("icf", "k")]
    checks += [("HMNEI", f"eps={e:g}", [("Epsilon", e)], lambda e=e: B.hmnei(X, y, epsilon=e))
               for e in grid_values("hmnei", "epsilon")]
    checks += [("IB3", f"{a:g}/{d:g}", [("Random Seed", B.KEEL_SEED), ("Acceptance Level", a), ("Drop Level", d)],
                lambda a=a, d=d: ib3(a, d)) for a, d in zip(grid_values("ib3", "accept"), grid_values("ib3", "drop"))]
    out = []
    for algo, label, params, ours in checks:
        rows, labels = run_keel(build, algo, X, y, params)
        out.append(f"{name} fold {fold} {algo} {label}: {compare(as_multiset(X, y, ours()), rows_to_multiset(X, y, rows, labels))}")
    return out


# ---- DROP3 ---------------------------------------------------------------------------------------------------------

def drop3_naive(X, y, k=3):
    """DROP3 with neighbor lists recomputed from scratch at every step (the reference for baselines.drop3).

    Args:
        X: (n, d) scaled training features.
        y: (n,) class labels.
        k: Number of neighbors.

    Returns:
        Sorted integer array of row indices into X of the retained instances.
    """
    y, n_classes = B._encode(y)
    T = np.flatnonzero(B._enn_mask(cdist(X, X), y, n_classes, k))
    yt = y[T]
    m = len(T)
    D = cdist(X[T], X[T])
    enemy = np.array([D[i, yt != yt[i]].min() if (yt != yt[i]).any() else np.inf for i in range(m)])
    visit = np.argsort(-enemy, kind="stable")
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")[:, : m - 1]
    alive = np.ones(m, dtype=bool)
    for p in visit:
        n_with = n_without = 0
        for a in range(m):
            nb = [b for b in order[a] if alive[b]][: k + 1]
            if p not in nb:
                continue
            n_with += B._vote([yt[b] for b in nb[:k]]) == yt[a]
            rest = [yt[b] for b in nb if b != p][:k]
            n_without += bool(rest) and B._vote(rest) == yt[a]
        if n_without >= n_with:
            alive[p] = False
    return T[alive]


def waashk_drop3():
    """waashk's DROP3 class, importable under current scikit-learn, and a copy with its departures from Wilson and
    Martinez (2000) undone. The two differ only in the four overrides of Patched.

    Fetches waashk/instanceselection at its pinned commit and shims the old scikit-learn imports it needs.

    Returns:
        Tuple (original, patched) of functions with signature (X, y, k) -> integer array of selected row indices.
    """
    import scipy.sparse
    import six
    import sklearn.externals
    import sklearn.neighbors
    root = fetch(WAASHK, ["src", "__init__.py"])
    sklearn.externals.six = six
    sys.modules["sklearn.externals.six"] = six
    sys.modules["sklearn.neighbors.classification"] = sklearn.neighbors
    sys.path.insert(0, str(root))
    from src.main.python.iSel import drop3 as mod

    class StableNumpy:
        """numpy for the patched module, with stable argsorts."""

        def __getattr__(self, name):
            """Forward every other attribute to numpy."""
            return getattr(np, name)

        @staticmethod
        def argsort(a, axis=-1):
            """np.argsort with kind="stable", so that ties keep the smaller index first."""
            return np.argsort(a, axis=axis, kind="stable")

    stable_np = StableNumpy()

    class Patched(mod.DROP3):
        """waashk's DROP3 with its four departures from Wilson and Martinez (2000) undone, see the overrides."""

        def ennpadrao(self, X, y):
            """(1) Leave-one-out ENN with the votes of baselines.enn; the original lets each point vote for itself.

            Args:
                X: Sparse (n, d) training features.
                y: (n,) class codes.

            Returns:
                Integer array of the row indices ENN keeps.
            """
            Xd = X.toarray()
            return np.flatnonzero(B._enn_mask(cdist(Xd, Xd), y, int(y.max()) + 1, self.n_neighbors))

        def set_nn(self, X, y):
            """(2) Exact distances, stable neighbor order, and k + 1 neighbors per point, so that the associates of
            P are the points with P among their k + 1 nearest neighbors.

            Args:
                X: Sparse (n, d) training features.
                y: (n,) class codes (unused).

            Returns:
                None; sets pairwise_distances, nn_complete and nn on the instance.
            """
            Xd = X.toarray()
            self.pairwise_distances = cdist(Xd, Xd)
            D = self.pairwise_distances.copy()
            np.fill_diagonal(D, -1.0)
            self.nn_complete = np.argsort(D, axis=1, kind="stable")[:, 1:]
            self.nn = [list(x[: self.n_neighbors + 1]) for x in self.nn_complete]

        def setMinEnemyDist(self, y):
            """(3) Visit by distance to the nearest enemy, farthest first (the original visits the nearest first);
            the stable argsort of StableNumpy keeps the smaller index first on ties.

            Args:
                y: (n,) class codes.

            Returns:
                None; sets mindist_ne (negated nearest-enemy distances) on the instance.
            """
            D = np.where(y[:, None] != y[None, :], self.pairwise_distances, np.inf)
            self.mindist_ne = -D.min(axis=1)

        def most_common(self, lst):
            """(4) Vote ties go to the tied class whose member is nearest (the original takes the smaller label).

            Args:
                lst: Labels ordered by increasing distance.

            Returns:
                The winning label.
            """
            return B._vote(list(lst))

    def run(cls, X, y, k):
        """Run one DROP3 class of waashk on dense data, silencing its prints.

        Args:
            cls: mod.DROP3 or Patched.
            X: (n, d) scaled training features.
            y: (n,) class codes.
            k: Number of neighbors.

        Returns:
            Integer array of the selected row indices.
        """
        m = cls(types.SimpleNamespace(outputdir=None), 0, n_neighbors=k)
        with contextlib.redirect_stdout(io.StringIO()):
            m.select_data(scipy.sparse.csr_matrix(X), y)
        return np.asarray(m.sample_indices_, dtype=int)

    def original(X, y, k):
        """waashk's DROP3 as is; see run for the arguments and the result."""
        return run(mod.DROP3, X, y, k)

    def patched(X, y, k):
        """Patched DROP3 with stable argsorts installed in the module for the call; see run for arguments and result."""
        mod.np = stable_np
        try:
            return run(Patched, X, y, k)
        finally:
            mod.np = np

    return original, patched


def drop3_task(args):
    """Compare baselines.drop3 on one fold with drop3_naive and with waashk's DROP3, as is and with (1)-(4) undone.

    Args:
        args: Tuple (dataset name, fold index, scaled training features, class codes).

    Returns:
        List of result lines, one per k of the drop3 grid, with the naive verdict, the size and Jaccard overlap of
        waashk's selection as is, and the verdict against the patched version.
    """
    name, fold, X, y = args
    original, patched = waashk_drop3()
    out = []
    for k in grid_values("drop3", "k"):
        ours = B.drop3(X, y, k=k)
        a, w = as_multiset(X, y, ours), original(X, y, k)
        jac = len(set(ours) & set(w)) / max(1, len(set(ours) | set(w)))
        out.append(f"{name} fold {fold} DROP3 k={k}: naive {compare(a, as_multiset(X, y, drop3_naive(X, y, k)))}; "
                   f"waashk as is |S|={len(w)} vs ours {len(ours)}, Jaccard {jac:.3f}; "
                   f"waashk with (1)-(4) undone {compare(a, as_multiset(X, y, patched(X, y, k)))}")
    return out


# ---- LSBo ----------------------------------------------------------------------------------------------------------

def lsbo_task(args):
    """Compare baselines.lsbo on one fold with dpr1005's LSBo, with a stable sort installed for ties in local-set size.

    Args:
        args: Tuple (dataset name, fold index, scaled training features, class codes).

    Returns:
        List with one result line ending with compare's verdict.
    """
    name, fold, X, y = args
    import pandas as pd
    root = fetch(DPR, ["instance_selection"])
    sys.path.insert(0, str(root))
    from instance_selection._LocalSets import LSBo, LocalSets
    # dpr1005 sorts by local-set size with pandas' default quicksort, which orders ties arbitrarily. With a stable
    # sort ties keep the index order, as in baselines.lsbo.
    LocalSets._sort_asc_lsc = lambda self: setattr(self, "local_sets", self.local_sets.sort_values(by="LSC", kind="stable"))
    xs, ys = LSBo().filter(pd.DataFrame(X), pd.DataFrame(y))
    ref = rows_to_multiset(X, y, xs.to_numpy(dtype=float), ys.to_numpy().ravel().astype(int), tol=1e-12)
    return [f"{name} fold {fold} LSBo: {compare(as_multiset(X, y, B.lsbo(X, y)), ref)}"]


# ---- PS (Bien and Tibshirani) ---------------------------------------------------------------------------------------

def protoclass_lib():
    """Install protoclass 1.0 from the CRAN mirror on GitHub into .cache/R once (needs R with the package class).

    Returns:
        The R library path .cache/R as a Path.
    """
    lib = CACHE / "R"
    if not (lib / "protoclass").exists():
        src = fetch(PROTOCLASS, ["DESCRIPTION", "NAMESPACE", "R", "man"])
        lib.mkdir(parents=True, exist_ok=True)
        subprocess.run(["R", "CMD", "INSTALL", "-l", str(lib), str(src)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return lib


def run_protoclass(D, y, eps_values):
    """Run protoclass's greedy (lambda = 1/n) in Rscript on the given distances, once per eps.

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        eps_values: Ball radii to run.

    Returns:
        List, one entry per eps, of sets of (point index, class code) pairs with positive alpha.
    """
    lib = protoclass_lib()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        D.astype("<f8").tofile(tmp / "d.bin")
        np.asarray(y, dtype="<i4").tofile(tmp / "y.bin")
        np.asarray(eps_values, dtype="<f8").tofile(tmp / "eps.bin")
        n, k = len(y), len(eps_values)
        script = f"""
            suppressMessages(library(protoclass, lib.loc="{lib}"))
            n <- {n}
            d <- matrix(readBin("{tmp / 'd.bin'}", "double", n * n, endian="little"), n, n)
            y <- readBin("{tmp / 'y.bin'}", "integer", n, endian="little")
            eps <- readBin("{tmp / 'eps.bin'}", "double", {k}, endian="little")
            for (e in seq_along(eps)) {{
                p <- protoclass(dxz=d, y=y, eps=eps[e], lambda=1/n)
                w <- which(p$alpha > 0, arr.ind=TRUE)
                cat(e, paste(w[, 1] - 1, p$classes[w[, 2]], sep=":", collapse=" "), "\n")
            }}"""
        out = subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, text=True).stdout
    pairs = [set() for _ in eps_values]
    for line in out.strip().split("\n"):
        e, *items = line.split()
        pairs[int(e) - 1] = {tuple(int(v) for v in item.split(":")) for item in items}
    return pairs


def ps_task(args):
    """Compare baselines.ps on one fold with protoclass at every q of its grid.

    The unrestricted greedy (every point a candidate for every class) is compared with protoclass; the own-class
    version that baselines.ps uses is compared with the unrestricted one, and checked to equal baselines.ps.

    Args:
        args: Tuple (dataset name, fold index, scaled training features, class codes).

    Returns:
        List of result lines, one per q.

    Raises:
        AssertionError: If _ps_greedy and ps disagree.
    """
    name, fold, X, y = args
    D = cdist(X, X)
    upper = squareform(D, checks=False)
    qs = grid_values("ps", "q")
    eps_values = [float(np.quantile(upper, q)) for q in qs]
    ref = run_protoclass(D, y, eps_values)
    out = []
    for q, eps, r in zip(qs, eps_values, ref):
        paper = set(B._ps_greedy(D, y, eps, own_class_only=False))
        ours = B._ps_greedy(D, y, eps)
        assert sorted(j for j, _ in ours) == B.ps(X, y, q=q).tolist()
        other = sum(c != y[j] for j, c in paper)
        same = {j for j, _ in ours} == {j for j, _ in paper} and other == 0
        out.append(f"{name} fold {fold} PS q={q:g}: unrestricted greedy vs protoclass "
                   f"{'identical' if paper == r else 'DIFFERENT'}; own-class version "
                   f"{'same selection' if same else f'differs ({len(ours)} vs {len(paper)} prototypes, {other} of another class)'}")
    return out


# ---- GKN (Gottlieb, Kontorovich and Nisnevitch) -------------------------------------------------------------------

def gkn_literal(D, y, removed):
    """Algorithms 1 and 2 of Gottlieb, Kontorovich and Nisnevitch (2014), transcribed line by line.

    Runs on the sample S left after removing `removed`, with the margin and diameter of S computed with numpy.

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        removed: Integer array of the row indices removed by the vertex cover.

    Returns:
        Tuple (S, margin, first, net): the remaining indices, the margin of S, the net after Algorithm 1 and the
        sorted net after Algorithm 2.
    """
    gone = set(removed.tolist())
    S = [i for i in range(len(y)) if i not in gone]
    DS, yS = D[np.ix_(S, S)], y[S]
    opp = yS[:, None] != yS[None, :]
    margin = float(DS[opp].min()) if opp.any() else np.inf          # definitions, computed with numpy
    diam = float(DS.max()) if len(S) > 1 else 0.0
    net = [S[0]]                                                     # Algorithm 1
    for p in S:
        if min(D[p, t] for t in net) >= margin:
            net.append(p)
    first = list(net)
    if np.isfinite(margin) and margin > 0:                          # Algorithm 2, distances scaled by diam(S)
        g = margin / diam
        alive = set(net)
        for i in range(0, int(np.ceil(np.log2(g))) - 1, -1):
            for p in net:
                if p not in alive:
                    continue
                if min((D[p, t] / diam for t in alive if y[t] != y[p]), default=np.inf) >= 2 * 2.0 ** i:
                    for t in [t for t in alive if t != p and D[p, t] / diam < 2.0 ** i - g]:
                        alive.discard(t)
        net = [p for p in net if p in alive]
    return S, margin, first, sorted(net)


def consistent(D, y, S, net):
    """Whether 1-NN on `net` labels every point of S correctly (strictly nearest net points; ties count as errors).

    Args:
        D: (n, n) pairwise distance matrix.
        y: (n,) class codes.
        S: Row indices of the sample.
        net: Row indices of the net.

    Returns:
        True if every point of S is labeled correctly, False otherwise.
    """
    for p in S:
        d = D[p, net]
        nearest = np.flatnonzero(d == d.min())
        if not all(y[net[j]] == y[p] for j in nearest):
            return False
    return True


def gkn_task(args):
    """Check baselines.gkn on one fold at every q of its grid.

    For each q it checks that the vertex cover is valid (no conflicting pair remains) and, for two classes, minimum
    (its size equals the maximum flow of the bipartite conflict graph, Konig, computed independently of the matching
    gkn uses); that gkn equals gkn_literal; and that the net is consistent with S after Algorithm 1 and after
    Algorithm 2, as the paper proves.

    Args:
        args: Tuple (dataset name, fold index, scaled training features, class codes).

    Returns:
        List of result lines, one per q.
    """
    name, fold, X, y = args
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    D = cdist(X, X)
    other = y[:, None] != y[None, :]
    nearest_enemy = np.where(other, D, np.inf).min(axis=1)
    out = []
    for q in grid_values("gkn", "q"):
        gamma = float(np.quantile(nearest_enemy, q))
        removed = B._min_vertex_cover(D, y, gamma)
        keep = np.setdiff1d(np.arange(len(y)), removed)
        valid = not ((D[np.ix_(keep, keep)] < gamma) & other[np.ix_(keep, keep)]).any()
        note = "nothing to remove" if len(removed) == 0 else "greedy, more than two classes"
        if len(np.unique(y)) == 2 and len(removed):
            # Minimal: the size of a minimum vertex cover of a bipartite graph is the maximum flow from one side to
            # the other (Konig), computed here independently of the matching used by gkn.
            L, R = np.flatnonzero(y == y.min()), np.flatnonzero(y != y.min())
            n = len(y)
            edges = np.argwhere((D[np.ix_(L, R)] < gamma))
            rows = np.concatenate([np.full(len(L), n), L[edges[:, 0]], R])
            cols = np.concatenate([L, R[edges[:, 1]], np.full(len(R), n + 1)])
            G = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n + 2, n + 2))
            flow = maximum_flow(G, n, n + 1).flow_value
            note = "minimum" if flow == len(removed) else f"NOT MINIMUM ({len(removed)} vs {flow})"
        S, margin, first, net = gkn_literal(D, y, removed)
        ours = B.gkn(X, y, q=q).tolist()
        out.append(f"{name} fold {fold} GKN q={q:g}: removed {len(removed)} ({'valid' if valid else 'INVALID'} cover, "
                   f"{note}); net {len(first)} -> pruned {len(net)}; literal transcription "
                   f"{'identical' if ours == net else 'DIFFERENT'}; consistent with S after Algorithm 1 "
                   f"{consistent(D, y, S, first)}, after Algorithm 2 {consistent(D, y, S, net)}")
    return out


def main():
    """Check the competitor selectors of baselines.py against reference implementations.

    Run (no environment variables needed):
        python verify_baselines.py keel  [dataset ...]   # enn, cnn, icf, hmnei and ib3 against KEEL (Java)
        python verify_baselines.py drop3 [dataset ...]   # drop3 against a naive version and waashk/instanceselection
        python verify_baselines.py lsbo  [dataset ...]   # lsbo against dpr1005
        python verify_baselines.py ps    [dataset ...]   # ps against protoclass (R), Bien and Tibshirani's package
        python verify_baselines.py gkn   [dataset ...]   # gkn against a literal transcription of its two algorithms
    Every check runs on the min-max scaled training part of each outer fold, as in run_real.py, in a pool of 4
    processes. The reference code is fetched with git into .cache/ at pinned commits (KEEL, WAASHK, DPR,
    PROTOCLASS), KEEL is compiled with javac (needs a JDK), and protoclass is installed from the CRAN mirror on
    GitHub into .cache/R (needs R with the recommended package class). Selections are compared as multisets of
    distinct training rows, because KEEL writes the selected rows rather than their indices (rounded, so they are
    matched to the nearest training row of the same class) and exact duplicates cannot be told apart.

    Args:
        sys.argv[1]: The check: keel, drop3, lsbo, ps or gkn.
        sys.argv[2:]: Optional dataset names; by default all 18 datasets for keel and, for the pure-Python
            references (quadratic in Python loops), the datasets below 2000 rows (SMALL).

    Returns:
        None. Appends one line per fold and configuration to results/verify_<check>.txt and prints it.
    """
    check, names = sys.argv[1], sys.argv[2:]
    if check == "keel":
        build = keel_build()
        tasks = [(n, f, X, y, build) for n in (names or DATASETS) for f, X, y in folds(n)]
        fn = keel_task
    else:
        tasks = [(n, f, X, y) for n in (names or SMALL) for f, X, y in folds(n)]
        fn = {"drop3": drop3_task, "lsbo": lsbo_task, "ps": ps_task, "gkn": gkn_task}[check]
        if check == "drop3":
            fetch(WAASHK, ["src", "__init__.py"])
        elif check == "lsbo":
            fetch(DPR, ["instance_selection"])
        elif check == "ps":
            protoclass_lib()
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"verify_{check}.txt", "a") as log, Pool(4, maxtasksperchild=1) as pool:
        for lines in pool.imap(fn, tasks):
            for line in lines:
                print(line, flush=True)
                log.write(line + "\n")


if __name__ == "__main__":
    main()

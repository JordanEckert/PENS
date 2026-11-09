from pathlib import Path

import gudhi
import numpy as np
import ripser
from scipy.spatial.distance import cdist, pdist, squareform

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def maxdim(truth):
    """Highest homology degree computed for a class: 2 if its beta2 > 0, else 1.

    Args:
        truth: (beta0, beta1, beta2) of the class, an entry of the truth dict of the generators.

    Returns:
        2 if truth[2] > 0, else 1.
    """
    return 2 if truth[2] > 0 else 1


def diagrams(pts, dim):
    """Finite-bar Vietoris-Rips persistence diagrams [H0, ..., H_dim] of a point set.

    The convention is that of pens.verify_certificate, shared by every synthetic script: Rips persistence by ripser
    from the full Euclidean distance matrix (an edge enters at its length), with the bars of infinite death (the
    essential degree-0 class) dropped.

    Args:
        pts: (m, d) array of points.
        dim: Highest homology degree computed.

    Returns:
        List of dim + 1 arrays of shape (n_bars, 2) with columns (birth, death); all empty when m < 2.
    """
    if len(pts) < 2:
        return [np.empty((0, 2)) for _ in range(dim + 1)]
    dgms = ripser.ripser(squareform(pdist(pts)), distance_matrix=True, maxdim=dim)["dgms"]
    return [d[np.isfinite(d[:, 1])] if len(d) else np.empty((0, 2)) for d in dgms]


def bott(d1, d2):
    """Bottleneck distance between two persistence diagrams, computed by GUDHI.

    Args:
        d1: (n1, 2) array of finite bars (birth, death); may be empty.
        d2: (n2, 2) array of finite bars (birth, death); may be empty.

    Returns:
        The bottleneck distance as a float (0.0 when both diagrams are empty).
    """
    return float(gudhi.bottleneck_distance(d1, d2))


def top_lifetimes(dgm, k=2):
    """The k longest lifetimes (death - birth) of a diagram, in decreasing order, padded with zeros.

    Args:
        dgm: (n, 2) array of finite bars (birth, death).
        k: Number of lifetimes returned.

    Returns:
        List of k floats, 0.0 in the positions beyond the number of bars.
    """
    life = np.sort(dgm[:, 1] - dgm[:, 0])[::-1]
    return [float(life[i]) if i < len(life) else 0.0 for i in range(k)]


def n_longer(dgm, threshold):
    """Number of bars with lifetime above threshold.

    Args:
        dgm: (n, 2) array of finite bars (birth, death).
        threshold: Lifetime a bar must exceed (strictly) to be counted.

    Returns:
        The count as an int.
    """
    return int(np.sum(dgm[:, 1] - dgm[:, 0] > threshold))


def fps(pts, k):
    """Greedy farthest-point sampling seeded at the point nearest the mean, stopped at k points.

    The eps-net construction of PENS (pens.net.farthest_point_cover) run to a fixed size instead of a radius; the
    matched-size farthest-point baseline of run_certificate.py (dB*_FX) and run_matched.py ("fps").

    Args:
        pts: (m, d) array of points, m >= 1.
        k: Number of points to select, at most m.

    Returns:
        Array of k local indices into pts, in selection order.
    """
    chosen = [int(np.argmin(cdist(pts, pts.mean(axis=0, keepdims=True)).ravel()))]
    mind = cdist(pts, pts[chosen]).ravel()
    while len(chosen) < k:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        np.minimum(mind, cdist(pts, pts[[nxt]]).ravel(), out=mind)
    return np.array(chosen)


class DiagramCache:
    """Diagrams keyed by the exact point set, so a set that recurs is computed once.

    The kept set across configurations, or a kept set equal to the whole class, recurs often in the sweeps. The key
    is the sorted index array (as bytes) together with the degree.

    Attributes:
        X: (n, d) array of all points; subsets are addressed by row index.
        store: Dict (sorted indices as bytes, dim) -> diagram list.
    """

    def __init__(self, X):
        """Create an empty cache over the rows of X.

        Args:
            X: (n, d) array of all points.

        Returns:
            None.
        """
        self.X = X
        self.store = {}

    def __call__(self, idx, dim):
        """Diagrams of X[idx] up to degree dim, from the cache when the same set and degree were requested before.

        Args:
            idx: Row indices of the point set (their order is irrelevant).
            dim: Highest homology degree.

        Returns:
            The list returned by diagrams(X[sorted idx], dim).
        """
        key = (np.sort(np.asarray(idx)).tobytes(), dim)
        if key not in self.store:
            self.store[key] = diagrams(self.X[np.sort(np.asarray(idx))], dim)
        return self.store[key]

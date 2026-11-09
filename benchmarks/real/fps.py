import numpy as np
from scipy.spatial.distance import cdist


def fps_class(pts, k):
    """Choose k rows of pts by farthest-point sampling seeded at the row nearest the mean.

    The rule is the one of benchmarks/synthetic/topology.fps: start at the row of pts nearest the mean of pts, then
    repeatedly add the row farthest from the rows chosen so far, until k rows are chosen. Ties go to the smaller row
    index (np.argmin and np.argmax return the first extremum).

    Args:
        pts: (m, d) array of the points of one class.
        k: Number of rows to choose, 1 <= k <= m.

    Returns:
        (k,) integer array of local row indices into pts, in selection order.

    Raises:
        AssertionError: If k is not in [1, m].
    """
    assert 1 <= k <= len(pts), (k, len(pts))
    chosen = [int(np.argmin(cdist(pts, pts.mean(axis=0, keepdims=True)).ravel()))]
    mind = cdist(pts, pts[chosen]).ravel()
    while len(chosen) < k:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        np.minimum(mind, cdist(pts, pts[[nxt]]).ravel(), out=mind)
    return np.array(chosen)


def fps_matched(X, y, counts):
    """Select counts[c] rows of each class c by farthest-point sampling within the class.

    This is the size-matched reference without edit or completion, used by run_components.py. The rows of a class
    are visited in increasing row order, so a tie inside fps_class resolves to the smaller row index of X.

    Args:
        X: (n, d) feature array.
        y: (n,) class labels.
        counts: Dict mapping class label c to the number of rows of class c to choose.

    Returns:
        Sorted integer array of row indices into X, with counts[c] rows of each class c.
    """
    parts = []
    for c, kc in counts.items():
        idx = np.flatnonzero(y == c)
        parts.append(idx[fps_class(X[idx], kc)])
    return np.sort(np.concatenate(parts))

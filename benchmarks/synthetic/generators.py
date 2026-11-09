import numpy as np


def _circle(rng, n, r=1.0, centre=(0.0, 0.0)):
    """Sample n points uniformly in angle on a circle, without noise.

    Args:
        rng: numpy Generator.
        n: Number of points.
        r: Radius.
        centre: (x, y) centre of the circle.

    Returns:
        (n, 2) array of points on the circle.
    """
    t = rng.uniform(0.0, 2 * np.pi, n)
    return np.column_stack([centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)])


def two_circles(seed, n=400):
    """Two concentric noisy circles, radii 1 (class 0) and 2 (class 1).

    Like every generator of this module it returns (X, y, truth) with truth = {class: (beta0, beta1, beta2)}, the
    Betti numbers of the clean shape of each class; coordinates are not normalized.

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 2) array with isotropic Gaussian noise of standard deviation 0.05; y, labels 0 then 1; truth
        {0: (1, 1, 0), 1: (1, 1, 0)}.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([_circle(rng, n, 1.0), _circle(rng, n, 2.0)])
    X = X + rng.normal(0.0, 0.05, X.shape)
    return X, np.repeat([0, 1], n), {0: (1, 1, 0), 1: (1, 1, 0)}


def circle_vs_disk(seed, n=400):
    """A noisy unit circle (class 0) around a filled disk of radius 0.9 (class 1) with the same centre.

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 2) array: the circle with Gaussian noise of standard deviation 0.10 and the disk sampled uniformly
        by area (radius 0.9 sqrt(U), U uniform on [0, 1]); y, labels 0 then 1; truth {0: (1, 1, 0), 1: (1, 0, 0)}.
    """
    rng = np.random.default_rng(seed)
    c = _circle(rng, n, 1.0) + rng.normal(0.0, 0.10, (n, 2))
    rad = 0.9 * np.sqrt(rng.uniform(0.0, 1.0, n))  # uniform on the disk
    ang = rng.uniform(0.0, 2 * np.pi, n)
    d = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    return np.vstack([c, d]), np.repeat([0, 1], n), {0: (1, 1, 0), 1: (1, 0, 0)}


def two_moons(seed, n=400, noise=0.15):
    """Two interleaved noisy half-moons, the construction of examples/basic_usage.py (one angle t shared by both).

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.
        noise: Standard deviation of the isotropic Gaussian noise.

    Returns:
        X, (2n, 2) array: the upper moon (cos t, sin t) and the lower moon (1 - cos t, 0.5 - sin t) for t uniform
        on [0, pi], plus noise; y, labels 0 then 1; truth {0: (1, 0, 0), 1: (1, 0, 0)}.
    """
    # same construction as PENS/examples/basic_usage.py (one t shared by both moons)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, n)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([upper, lower]) + rng.normal(0.0, noise, (2 * n, 2))
    return X, np.repeat([0, 1], n), {0: (1, 0, 0), 1: (1, 0, 0)}


def crossing_circles(seed, n=400):
    """Two noisy unit circles whose centres are 1 apart, at (-0.5, 0) (class 0) and (0.5, 0) (class 1).

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 2) array with isotropic Gaussian noise of standard deviation 0.05; y, labels 0 then 1; truth
        {0: (1, 1, 0), 1: (1, 1, 0)}.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([_circle(rng, n, 1.0, (-0.5, 0.0)), _circle(rng, n, 1.0, (0.5, 0.0))])
    X = X + rng.normal(0.0, 0.05, X.shape)
    return X, np.repeat([0, 1], n), {0: (1, 1, 0), 1: (1, 1, 0)}


def torus_blob(seed, n_torus=800, n_blob=400, R=2.9, r=1.2):
    """A torus (class 0), sampled uniformly by area, around a Gaussian blob (class 1) in its hole.

    The tube is thick enough (r = 1.2) and sampled densely enough that the void is the one long degree-2 bar, and
    the hole (radius R - r = 1.7) keeps the blob apart from the torus. The angle around the tube has density
    proportional to R + r cos(ph) and is drawn by rejection.

    Args:
        seed: Seed of the numpy Generator.
        n_torus: Points on the torus.
        n_blob: Points in the blob.
        R: Distance from the centre of the hole to the centre of the tube.
        r: Radius of the tube.

    Returns:
        X, (n_torus + n_blob, 3) array: the torus with isotropic Gaussian noise of standard deviation 0.03 and the
        blob N(0, 0.3^2 I); y, labels 0 then 1; truth {0: (1, 2, 1), 1: (1, 0, 0)}.
    """
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, 2 * np.pi, n_torus)
    # The angle around the tube has density proportional to R + r cos(ph); draw it by rejection.
    ph = np.empty(0)
    while len(ph) < n_torus:
        p = rng.uniform(0.0, 2 * np.pi, n_torus)
        ph = np.concatenate([ph, p[rng.uniform(0.0, R + r, n_torus) < R + r * np.cos(p)]])
    ph = ph[:n_torus]
    T = np.column_stack([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th), r * np.sin(ph)])
    T = T + rng.normal(0.0, 0.03, T.shape)
    B = rng.normal(0.0, 0.3, (n_blob, 3))
    y = np.concatenate([np.zeros(n_torus, dtype=int), np.ones(n_blob, dtype=int)])
    return np.vstack([T, B]), y, {0: (1, 2, 1), 1: (1, 0, 0)}


def circles_r10(seed, n=400):
    """Two unit circles in R^10 in disjoint coordinate planes: coordinates (0, 1) for class 0, (2, 3) for class 1.

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 10) array, zero outside the two planes before isotropic Gaussian noise of standard deviation 0.05
        is added to all ten coordinates; y, labels 0 then 1; truth {0: (1, 1, 0), 1: (1, 1, 0)}.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((2 * n, 10))
    X[:n, 0:2] = _circle(rng, n, 1.0)
    X[n:, 2:4] = _circle(rng, n, 1.0)
    X = X + rng.normal(0.0, 0.05, X.shape)
    return X, np.repeat([0, 1], n), {0: (1, 1, 0), 1: (1, 1, 0)}


def _gauss(d):
    """Build the generator of two standard normal classes in R^d whose means differ by 3.29 along the first axis.

    Args:
        d: Dimension.

    Returns:
        The generator function gen(seed, n=400), named gauss_d{d}; see gen.
    """
    def gen(seed, n=400):
        """Draw two standard normal classes in R^d, class 1 shifted by 3.29 along the first coordinate.

        The shift is 2 * 1.645 standard deviations, so the Bayes error of the two-class problem is about 5%.

        Args:
            seed: Seed of the numpy Generator.
            n: Points per class.

        Returns:
            X, (2n, d) array; y, labels 0 then 1; truth {0: (1, 0, 0), 1: (1, 0, 0)}.
        """
        rng = np.random.default_rng(seed)
        X = rng.normal(0.0, 1.0, (2 * n, d))
        X[n:, 0] += 3.29
        return X, np.repeat([0, 1], n), {0: (1, 0, 0), 1: (1, 0, 0)}
    gen.__name__ = f"gauss_d{d}"
    return gen


def _sphere(rng, n, r=1.0):
    """Sample n points uniformly on a sphere in R^3 (normalized Gaussian vectors), without noise.

    Args:
        rng: numpy Generator.
        n: Number of points.
        r: Radius.

    Returns:
        (n, 3) array of points on the sphere.
    """
    v = rng.normal(size=(n, 3))
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def sphere_blob(seed, n=300):
    """A noisy unit sphere (class 0, beta2 = 1) around a Gaussian blob (class 1) at its centre.

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 3) array: the sphere with isotropic Gaussian noise of standard deviation 0.03 and the blob
        N(0, 0.25^2 I); y, labels 0 then 1; truth {0: (1, 0, 1), 1: (1, 0, 0)}.
    """
    rng = np.random.default_rng(seed)
    S = _sphere(rng, n) + rng.normal(0.0, 0.03, (n, 3))
    B = rng.normal(0.0, 0.25, (n, 3))
    return np.vstack([S, B]), np.repeat([0, 1], n), {0: (1, 0, 1), 1: (1, 0, 0)}


def two_spheres(seed, n=300):
    """Two concentric noisy spheres, radii 1 (class 0) and 2 (class 1).

    Args:
        seed: Seed of the numpy Generator.
        n: Points per class.

    Returns:
        X, (2n, 3) array with isotropic Gaussian noise of standard deviation 0.03; y, labels 0 then 1; truth
        {0: (1, 0, 1), 1: (1, 0, 1)}.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([_sphere(rng, n, 1.0), _sphere(rng, n, 2.0)]) + rng.normal(0.0, 0.03, (2 * n, 3))
    return X, np.repeat([0, 1], n), {0: (1, 0, 1), 1: (1, 0, 1)}


def _embedded(base, D, noise=0.02):
    """Build the generator that maps `base` isometrically into R^D by a random orthonormal frame and adds noise.

    Gaussian noise of standard deviation `noise` is added in every one of the D coordinates. The frame is that of
    the training draw at seed % 1000, so that the fresh draw at seed + 1000 (the clean test set of run_ablation.py
    and run_matched.py) lies in the same subspace as the training draw at seed; the draws at seeds below 1000 are
    unchanged.

    Args:
        base: A generator of this module, called as base(seed) or base(seed, n).
        D: Ambient dimension, at least the dimension of the output of base.
        noise: Standard deviation of the isotropic Gaussian noise added after the embedding.

    Returns:
        The generator function gen(seed, n=None), named {base.__name__}_D{D}; see gen.
    """
    def gen(seed, n=None):
        """Draw `base` at `seed` and embed it in R^D.

        Args:
            seed: Seed; the frame is drawn from default_rng([seed % 1000, D]) and the noise from default_rng([seed, D]).
            n: Points per class passed on to base, or None for the default of base.

        Returns:
            X, (len(X), D) array; y and truth as returned by base (the Betti numbers are unchanged by the isometry).
        """
        X, y, truth = base(seed) if n is None else base(seed, n)
        rng = np.random.default_rng([seed, D])
        frame = rng if seed < 1000 else np.random.default_rng([seed % 1000, D])
        Q, _ = np.linalg.qr(frame.normal(size=(D, X.shape[1])))
        return X @ Q.T + rng.normal(0.0, noise, (len(X), D)), y, truth
    gen.__name__ = f"{base.__name__}_D{D}"
    return gen


# Name -> generator. Each is called as gen(seed) or gen(seed, n) and returns (X, y, truth), truth = {class: (beta0,
# beta1, beta2)} of the clean shape of each class; coordinates are not normalized. Betti numbers, class 0 | class 1:
#   two_circles, crossing_circles, circles_r10, two_circles_D*   (1, 1, 0) | (1, 1, 0)
#   circle_vs_disk                                                (1, 1, 0) | (1, 0, 0)
#   two_moons, gauss_d*                                           (1, 0, 0) | (1, 0, 0)
#   torus_blob                                                    (1, 2, 1) | (1, 0, 0)
#   sphere_blob, sphere_blob_D*                                   (1, 0, 1) | (1, 0, 0)
#   two_spheres                                                   (1, 0, 1) | (1, 0, 1)
GENERATORS = {
    "two_circles": two_circles,
    "circle_vs_disk": circle_vs_disk,
    "two_moons": two_moons,
    "crossing_circles": crossing_circles,
    "torus_blob": torus_blob,
    "circles_r10": circles_r10,
    "gauss_d2": _gauss(2),
    "gauss_d5": _gauss(5),
    "gauss_d10": _gauss(10),
    "gauss_d20": _gauss(20),
    "sphere_blob": sphere_blob,
    "two_spheres": two_spheres,
    **{f"two_circles_D{D}": _embedded(two_circles, D) for D in (10, 50, 100)},
    **{f"sphere_blob_D{D}": _embedded(sphere_blob, D) for D in (10, 50, 100)},
}

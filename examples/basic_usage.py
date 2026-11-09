import numpy as np

from pens import PENS, bottleneck_bound, edit_cost, end_to_end_bound, verify_certificate


def two_moons(n_per=150, noise=0.08, seed=0):
    """Generate two interleaving half-circles with Gaussian noise.

    Args:
        n_per: Points per moon.
        noise: Standard deviation of the added noise.
        seed: Seed for numpy's default_rng.

    Returns:
        Tuple (X, y) with X of shape (2 * n_per, 2) and labels in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, n_per)
    upper = np.column_stack([np.cos(t), np.sin(t)])
    lower = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([upper, lower]) + rng.normal(0.0, noise, (2 * n_per, 2))
    y = np.repeat([0, 1], n_per)
    return X, y


def main():
    """Fit PENS at its defaults on two moons and print the fit, the certificate check and the end-to-end bound.

    Args:
        None; the dataset comes from `two_moons()`.

    Returns:
        None. Prints the prototype counts per class, the reduced set's shape, the Corollary 1 check of
        `verify_certificate` (skipped with a note when ripser or gudhi is missing) and the Theorem 3 bound.
    """
    X, y = two_moons()
    model = PENS().fit(X, y)

    print(f"{len(X)} points -> {len(model.prototype_indices_)} prototypes")
    bounds = bottleneck_bound(model.radii_)
    for label in model.classes_:
        n_c = int(np.sum(y == label))
        print(
            f"class {label}: {n_c} points, {len(model.kept_[label])} kept, "
            f"{len(model.core_[label])} core + {len(model.prototypes_[label]) - len(model.core_[label])} completion prototypes, "
            f"eps = {model.radii_[label]:.4f}, bottleneck bound = {bounds[label]:.4f}"
        )

    X_red, y_red = model.transform(X, y)
    print(f"reduced set: X {X_red.shape}, y {y_red.shape}")

    try:
        report = verify_certificate(X, model.kept_, model.prototypes_, model.radii_, maxdim=1)
    except ImportError as err:
        print(f"certificate check skipped: {err}")
        return
    for label, r in report.items():
        bars = ", ".join(f"H{k} {v:.4f}" for k, v in r["bottleneck"].items())
        print(
            f"class {label}: d_H = {r['hausdorff']:.4f} <= {r['eps']:.4f}; "
            f"d_B: {bars} <= {r['bound']:.4f}; holds = {r['holds']}"
        )
    cost = edit_cost(X, y, model.kept_)
    for label, b in end_to_end_bound(X, y, model.kept_, model.radii_).items():
        print(f"class {label}: edit cost d_H(K, X_c) = {cost[label]:.4f}; bound against the whole class = {b:.4f}")


if __name__ == "__main__":
    main()

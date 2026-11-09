from __future__ import annotations

from typing import Any

import numpy as np

from ._metrics import to_scipy_metric
from .completion import balanced_weights, consistent_completion, gain_completion, prune_completion
from .dendrogram import descend
from .net import centroid_nearest, farthest_point_cover, merge_scale

_CLASS_WEIGHTS = (None, "balanced")
_COMPLETIONS = (None, "consistent", "gain")


class PENS:
    """Supervised prototype selector that edits the pooled single-linkage dendrogram, covers each edited class with
    an eps-net, and completes the net at the class boundaries (Algorithm 1 of the paper).

    The eps-net (the core Q_c) bounds the bottleneck distance between the persistence diagrams of the prototypes
    and of the kept points of each class by 2 eps_c (Corollary 1). Completion only adds kept points and never
    removes a core point, so it leaves that bound unchanged. The fit is deterministic; every tie is broken by a
    fixed rule (see the README).

    Args:
        purity: Majority-class fraction at which a dendrogram node is accepted, in (0, 1]. The test is exact
            (rational arithmetic), so a node of purity exactly `purity` is accepted.
        eps_scale: Net radius in units of the median single-linkage merge height of the pooled kept set, > 0.
        s_min: Nodes of this size or smaller are accepted without further splitting, >= 1.
        min_keep: Accepted clusters keeping fewer points than this are dropped, >= 1.
        class_weight: "balanced" weights every class by the inverse of its size in the purity test, the
            majority claim, completion and pruning; None counts points.
        completion: None, "consistent" (add kept points until every kept point has a nearest prototype of its own
            class) or "gain" (add a kept point only where it corrects at least `gain` other kept points).
        gain: Threshold for completion="gain", in units of the mean point weight, >= 0.
        prune: Remove completion points whose removal does not lower the weighted number of correctly
            classified kept points. Core points are never removed.
        metric: SciPy metric name (the aliases manhattan/l1/l2 are accepted); stored in SciPy spelling.
        metric_params: Extra keyword arguments passed to scipy pdist/cdist, or None.

    Attributes:
        classes_: Sorted array of the class labels seen in fit.
        prototypes_: Dict mapping class label to the sorted row indices of its prototypes P_c.
        prototype_indices_: Sorted array of all prototype row indices.
        kept_: Dict mapping class label to the row indices of its kept set K_c (empty for a guard class).
        core_: Dict mapping class label to the sorted row indices of its eps-net core Q_c (the guard row for a
            guard class).
        radii_: Dict mapping class label to eps_c: the shared radius when |K_c| >= 2, else 0.0.
        clusters_: List of (class label, kept row indices) for the accepted clusters that survived min_keep.
    """

    def __init__(
        self,
        purity: float = 0.9,
        eps_scale: float = 4.0,
        s_min: int = 5,
        min_keep: int = 1,
        class_weight: str | None = "balanced",
        completion: str | None = "consistent",
        gain: float = 1.0,
        prune: bool = True,
        metric: str = "euclidean",
        metric_params: dict[str, Any] | None = None,
    ):
        if not 0 < purity <= 1:
            raise ValueError(f"purity must be in (0, 1], got {purity}")
        if not eps_scale > 0:
            raise ValueError(f"eps_scale must be > 0, got {eps_scale}")
        if s_min < 1:
            raise ValueError(f"s_min must be >= 1, got {s_min}")
        if min_keep < 1:
            raise ValueError(f"min_keep must be >= 1, got {min_keep}")
        if class_weight not in _CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {_CLASS_WEIGHTS}, got {class_weight!r}")
        if completion not in _COMPLETIONS:
            raise ValueError(f"completion must be one of {_COMPLETIONS}, got {completion!r}")
        if not gain >= 0:
            raise ValueError(f"gain must be >= 0, got {gain}")
        self.purity = purity
        self.eps_scale = eps_scale
        self.s_min = s_min
        self.min_keep = min_keep
        self.class_weight = class_weight
        self.completion = completion
        self.gain = gain
        self.prune = prune
        self.metric = to_scipy_metric(metric)
        self.metric_params = metric_params or {}

    def fit(self, X, y) -> PENS:
        """Select prototypes for every class of the training set and set the fitted attributes.

        Stage 1 edits the pooled single-linkage tree, Stage 2 sets the shared radius eps = eps_scale x median merge
        height of the pooled kept set and covers each kept set by farthest-point sampling (its core), and Stage 3
        runs the chosen completion and pruning. A class whose kept set is empty receives one guard prototype, the
        row of the class nearest its mean, with radius 0. With a single row (n = 1) that row is the only prototype;
        with no rows every attribute is empty.

        Args:
            X: (n_samples, n_features) feature array; cast to float64.
            y: (n_samples,) class labels of any hashable type.

        Returns:
            The fitted selector.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes, y_enc = np.unique(y, return_inverse=True)
        n, n_classes = len(X), len(classes)
        self.classes_ = classes
        self.prototypes_, self.kept_, self.radii_, self.core_, self.clusters_ = {}, {}, {}, {}, []

        if n == 1:
            single = np.array([0], dtype=int)
            self.prototypes_[classes[0]] = single
            self.kept_[classes[0]] = single
            self.core_[classes[0]] = single
            self.radii_[classes[0]] = 0.0
        elif n > 1:
            balanced = self.class_weight == "balanced"
            class_totals = np.bincount(y_enc, minlength=n_classes)
            clusters = descend(
                X, y_enc, n_classes, class_totals, self.purity, self.s_min,
                self.min_keep, self.metric, self.metric_params, balanced,
            )
            self.clusters_ = [(classes[c], kept) for c, kept in clusters]
            all_kept = np.sort(np.concatenate([m for _, m in clusters])) if clusters else np.empty(0, dtype=int)
            eps_all = self.eps_scale * merge_scale(X, all_kept, self.metric, self.metric_params)
            for c in range(n_classes):
                parts = [m for cc, m in clusters if cc == c]
                kept_c = np.concatenate(parts) if parts else np.empty(0, dtype=int)
                if len(kept_c) == 0:
                    members = np.flatnonzero(y_enc == c)
                    chosen = [int(members[centroid_nearest(X[members], self.metric, self.metric_params)])]
                    eps_c = 0.0
                else:
                    eps_c = eps_all if len(kept_c) >= 2 else 0.0
                    chosen = farthest_point_cover(X, np.sort(kept_c), eps_c, self.metric, self.metric_params)
                self.core_[classes[c]] = np.array(sorted(set(chosen)), dtype=int)
                self.kept_[classes[c]] = kept_c
                self.radii_[classes[c]] = eps_c

            S = sorted({int(i) for core in self.core_.values() for i in core})
            if self.completion is not None and len(all_kept):
                w = balanced_weights(y_enc[all_kept], n_classes) if balanced else np.ones(len(all_kept))
                if self.completion == "consistent":
                    S = consistent_completion(X, y_enc, all_kept, S, self.metric, self.metric_params)
                else:
                    S = gain_completion(X, y_enc, n_classes, all_kept, S, self.gain, w, self.metric, self.metric_params)
                S = sorted(set(S))
                if self.prune:
                    protected = {int(i) for core in self.core_.values() for i in core}
                    S = prune_completion(X, y_enc, all_kept, S, protected, w, self.metric, self.metric_params)
            S = np.array(S, dtype=int)
            for c in range(n_classes):
                self.prototypes_[classes[c]] = S[y_enc[S] == c]

        parts = list(self.prototypes_.values())
        self.prototype_indices_ = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=int)
        return self

    def select(self, X, y) -> dict:
        """Fit the selector and return the per-class prototypes.

        Args:
            X: (n_samples, n_features) feature array.
            y: (n_samples,) class labels.

        Returns:
            Dict mapping each class label to sorted original row indices.
        """
        return self.fit(X, y).prototypes_

    def transform(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Restrict a dataset to the fitted prototype rows.

        Args:
            X: (n_samples, n_features) feature array used in fit.
            y: (n_samples,) class labels used in fit.

        Returns:
            Tuple (X_prototypes, y_prototypes).
        """
        return np.asarray(X)[self.prototype_indices_], np.asarray(y)[self.prototype_indices_]

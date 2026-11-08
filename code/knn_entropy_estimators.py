#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-parametric k-nearest-neighbour differential entropy estimator.

Implements the Kozachenko-Leonenko (1987) k-NN differential entropy
estimator for continuous multivariate data, plus the same Total
Correlation / Dual Total Correlation / O-information combinators used
in ``entropy_estimators.py`` (the Gaussian-copula version), so the two
estimators can be compared using identical formulas on identical data.

Estimator (exact formula used here)
------------------------------------
For a sample of N i.i.d. draws of a d-dimensional continuous variable X::

    H_hat(X) = psi(N) - psi(k) + log(V_d) + (d / N) * sum_i log(r_{k,i})

where:
  - N is the number of observations, d the dimensionality of X,
  - r_{k,i} is the distance (under the metric below) from observation i
    to its k-th nearest neighbour among the OTHER N-1 observations
    (i.e. i itself is excluded),
  - V_d = pi^(d/2) / Gamma(d/2 + 1) is the volume of the d-dimensional
    Euclidean unit ball (radius 1),
  - psi is the digamma function (``scipy.special.digamma``).

This is the standard Kozachenko-Leonenko estimator as used in, e.g.,
Kraskov, Stogbauer & Grassberger (2004), "Estimating mutual information",
Phys. Rev. E 69, 066138, eq. (20), written here directly in terms of the
k-th-neighbour RADIUS r_{k,i} (not the diameter 2*r_{k,i}); the unit-ball
volume V_d above is defined consistently for the radius convention, so
no extra factor of 2^d appears.

Distance metric: Euclidean (L2) distance, via ``scipy.spatial.cKDTree``.

Treatment of ties: not specially handled. With continuous (absolutely
continuous) data, exact distance ties occur with probability zero; if
they do occur (e.g. discretized input), ``cKDTree`` resolves them
deterministically by internal index order, which can introduce a small,
unquantified bias. This module does not add tie-breaking jitter itself;
callers working with discrete or heavily-rounded data should add their
own small continuous jitter before calling these functions.

Selected k: not fixed by this module; passed explicitly by the caller.
``validate_higher_order_estimators.py`` uses k = 5 as the primary choice,
with k = 3 and k = 10 for a sensitivity check.

Random seed: this module performs no random data generation; the caller
is responsible for seeding any random draws before calling these
functions.

Units: nats (natural log) throughout, matching ``entropy_estimators.py``.

No additional Gaussian (or other distributional) assumption is made
anywhere in this module -- the estimator is fully non-parametric.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

logger = logging.getLogger(__name__)


def _unit_ball_log_volume(d: int) -> float:
    """log(V_d) for the d-dimensional Euclidean unit ball, V_d = pi^(d/2) / Gamma(d/2 + 1)."""
    return (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)


def _check_input(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
    n_obs = x.shape[0]
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/inf; the kNN entropy estimator requires finite data "
                          "(no filling/interpolation is performed here).")
    if k < 1 or k >= n_obs:
        raise ValueError(f"k must satisfy 1 <= k < n_obs; got k={k}, n_obs={n_obs}")
    return x


def knn_entropy(x: np.ndarray, k: int = 5, workers: int = 1) -> float:
    """Kozachenko-Leonenko k-NN differential entropy of X, in nats.

    Args:
        x: (n_obs, d) array of continuous observations (1-D input is
            treated as (n_obs, 1)). Must be finite (no NaN/inf).
        k: Number of nearest neighbours to use (1 <= k < n_obs).
        workers: Number of threads for the KD-tree query (``scipy``
            ``workers`` argument). Default 1 (single-threaded) for
            deterministic, reproducible timing; increase only if runtime
            matters more than that.

    Returns:
        Estimated differential entropy H(X), in nats.
    """
    x = _check_input(x, k)
    n_obs, d = x.shape
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=k + 1, p=2, workers=workers)  # index 0 is self (distance 0)
    r_k = dist[:, k]
    r_k = np.maximum(r_k, np.finfo(float).tiny)  # guard against log(0) in degenerate cases
    return float(digamma(n_obs) - digamma(k) + _unit_ball_log_volume(d) + (d / n_obs) * np.sum(np.log(r_k)))


def knn_entropy_multi_k(x: np.ndarray, k_list: Sequence[int], workers: int = 1) -> Dict[int, float]:
    """Compute :func:`knn_entropy` for several k values from one shared
    KD-tree build and one shared neighbour query (efficiency only --
    mathematically identical to calling ``knn_entropy`` separately for
    each k).

    Args:
        x: (n_obs, d) array, see :func:`knn_entropy`.
        k_list: Collection of k values to evaluate.
        workers: See :func:`knn_entropy`.

    Returns:
        Dict mapping each requested k to its entropy estimate (nats).
    """
    k_list = sorted(set(int(k) for k in k_list))
    k_max = k_list[-1]
    x = _check_input(x, k_max)
    n_obs, d = x.shape

    tree = cKDTree(x)
    dist, _ = tree.query(x, k=k_max + 1, p=2, workers=workers)
    log_vd = _unit_ball_log_volume(d)

    out: Dict[int, float] = {}
    for k in k_list:
        r_k = np.maximum(dist[:, k], np.finfo(float).tiny)
        out[k] = float(digamma(n_obs) - digamma(k) + log_vd + (d / n_obs) * np.sum(np.log(r_k)))
    return out


def knn_o_information_multi_k(
    x_full: np.ndarray, idx: Sequence[int], k_list: Sequence[int], workers: int = 1
) -> Dict[int, Tuple[float, float, float]]:
    """O-information (and its T, D components) of the variable subset
    ``idx`` of ``x_full``, for several k values, using the k-NN entropy
    estimator throughout.

        T(X_S) = sum_{i in S} H(X_i) - H(X_S)
        D(X_S) = sum_{i in S} H(X_{S\\{i}}) - (|S|-1) * H(X_S)
        Omega(X_S) = T(X_S) - D(X_S)

    KD-tree queries are shared across the requested k values for each
    distinct variable subset needed (marginals, the full joint subset,
    and each leave-one-out subset), so this is efficient for a
    k-sensitivity sweep.

    Args:
        x_full: (n_obs, n_vars) array of continuous observations
            (columns are the full variable set; ``idx`` selects the
            subset S).
        idx: Column indices (into ``x_full``) making up the subset S.
        k_list: k values to evaluate.
        workers: See :func:`knn_entropy`.

    Returns:
        Dict mapping each requested k to a tuple ``(omega, T, D)``, nats.
    """
    idx = list(idx)
    n = len(idx)
    k_list = sorted(set(int(k) for k in k_list))

    h_full_by_k = knn_entropy_multi_k(x_full[:, idx], k_list, workers=workers)

    h_marg_by_k = {k: 0.0 for k in k_list}
    for i in idx:
        hi = knn_entropy_multi_k(x_full[:, [i]], k_list, workers=workers)
        for k in k_list:
            h_marg_by_k[k] += hi[k]

    h_loo_by_k = {k: 0.0 for k in k_list}
    if n >= 2:
        for i in idx:
            rest = [j for j in idx if j != i]
            hr = knn_entropy_multi_k(x_full[:, rest], k_list, workers=workers)
            for k in k_list:
                h_loo_by_k[k] += hr[k]

    out: Dict[int, Tuple[float, float, float]] = {}
    for k in k_list:
        t = h_marg_by_k[k] - h_full_by_k[k]
        dd = (h_loo_by_k[k] - (n - 1) * h_full_by_k[k]) if n >= 2 else 0.0
        out[k] = (t - dd, t, dd)
    return out

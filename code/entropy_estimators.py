#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gaussian-copula-based entropy estimators for continuous multivariate data.

Implements the minimal building blocks needed for static O-information
analysis of continuous financial returns:

  - A rank-based Gaussian copula transform (empirical CDF -> standard
    normal scores), which Gaussianizes each variable's marginal while
    preserving its rank structure (its copula).
  - Closed-form multivariate Gaussian differential entropy from a
    correlation matrix.
  - Total Correlation T(X^n), Dual Total Correlation D(X^n), and
    O-information Omega(X^n) = T(X^n) - D(X^n), for an arbitrary subset
    of variables.

This module is a pure estimator library: it does not read or write any
project data file. All quantities are returned in nats (natural log).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm, rankdata

logger = logging.getLogger(__name__)

# Differential entropy of a univariate standard normal, in nats:
# H(N(0,1)) = 0.5 * log(2*pi*e)
LOG_2PI_E: float = float(np.log(2.0 * np.pi * np.e))


def gaussian_copula_transform(x: np.ndarray) -> np.ndarray:
    """Rank-transform each column of ``x`` to standard normal scores.

    For each variable (column), the empirical CDF is estimated by ranking
    the observations (average rank for ties) and mapping rank / (n + 1) to
    the open interval (0, 1); the inverse standard normal CDF then maps
    these empirical uniform scores to standard normal scores. This forces
    every variable's marginal to be standard normal while preserving its
    rank structure (its copula) -- the usual construction for a Gaussian
    copula entropy/mutual-information estimator.

    Args:
        x: (n_obs, n_vars) array of continuous observations. Must not
            contain NaN or infinite values.

    Returns:
        (n_obs, n_vars) array of Gaussianized scores.

    Raises:
        ValueError: If ``x`` is not 2-D, contains non-finite values, or has
            fewer than 2 observations.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_obs, n_vars), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "x contains NaN or infinite values; the Gaussian copula "
            "transform requires complete, finite data (no filling/"
            "interpolation is performed here)."
        )
    n_obs = x.shape[0]
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")

    ranks = np.apply_along_axis(rankdata, 0, x)  # average rank for ties, in [1, n_obs]
    u = ranks / (n_obs + 1.0)  # empirical uniform scores, strictly inside (0, 1)
    z = norm.ppf(u)
    return z


def correlation_matrix(z: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of Gaussianized scores ``z``.

    Args:
        z: (n_obs, n_vars) array, typically the output of
            :func:`gaussian_copula_transform`.

    Returns:
        (n_vars, n_vars) correlation matrix.

    Raises:
        ValueError: If ``z`` has fewer than 2 observations.
    """
    z = np.asarray(z, dtype=float)
    if z.shape[0] < 2:
        raise ValueError("Need at least 2 observations to estimate a correlation matrix.")
    r = np.corrcoef(z, rowvar=False)
    r = np.atleast_2d(r)
    return r


def gaussian_differential_entropy(r: np.ndarray, idx: Optional[Sequence[int]] = None) -> float:
    """Differential entropy (nats) of a standard-normal-marginal multivariate
    Gaussian with correlation matrix ``r``, restricted to the variable
    subset ``idx``.

    H(X_S) = 0.5 * |S| * log(2*pi*e) + 0.5 * log(det(R_S))

    where ``R_S`` is the correlation submatrix for the variables in ``idx``.

    Args:
        r: (d, d) correlation matrix of the full variable set.
        idx: Indices (into the rows/columns of ``r``) of the subset S;
            ``None`` means "all variables".

    Returns:
        Differential entropy of X_S, in nats. Returns 0.0 for an empty subset.

    Raises:
        numpy.linalg.LinAlgError: If the correlation submatrix for ``idx``
            is not positive definite.
    """
    r = np.asarray(r, dtype=float)
    if idx is None:
        r_sub = r
        d = r.shape[0]
    else:
        idx = list(idx)
        d = len(idx)
        if d == 0:
            return 0.0
        r_sub = r[np.ix_(idx, idx)]

    sign, logdet = np.linalg.slogdet(r_sub)
    if sign <= 0:
        raise np.linalg.LinAlgError(
            f"Correlation submatrix for idx={idx} is not positive definite "
            f"(sign={sign}); cannot compute a valid differential entropy."
        )
    return 0.5 * d * LOG_2PI_E + 0.5 * logdet


def entropy_marginal(r: np.ndarray, i: int) -> float:
    """H(X_i) for a single Gaussianized variable.

    Under the Gaussian copula construction this is always
    ``0.5 * log(2*pi*e)``, independent of ``i`` -- implemented via the
    generic subset formula so the same code path is exercised and tested.
    """
    return gaussian_differential_entropy(r, [i])


def total_correlation(r: np.ndarray, idx: Sequence[int]) -> float:
    """Total correlation (multi-information) of the subset ``idx``.

    T(X_S) = sum_{i in S} H(X_i) - H(X_S)
    """
    idx = list(idx)
    h_full = gaussian_differential_entropy(r, idx)
    h_sum_marginals = sum(entropy_marginal(r, i) for i in idx)
    return h_sum_marginals - h_full


def dual_total_correlation(r: np.ndarray, idx: Sequence[int]) -> float:
    """Dual total correlation of the subset ``idx``.

    D(X_S) = sum_{i in S} H(X_{S \\ {i}}) - (|S| - 1) * H(X_S)

    Returns 0.0 for |S| < 2 (dual total correlation is not meaningfully
    defined below two variables).
    """
    idx = list(idx)
    n = len(idx)
    if n < 2:
        return 0.0
    h_full = gaussian_differential_entropy(r, idx)
    h_sum_leave_one_out = 0.0
    for i in idx:
        rest = [j for j in idx if j != i]
        h_sum_leave_one_out += gaussian_differential_entropy(r, rest)
    return h_sum_leave_one_out - (n - 1) * h_full


def o_information(r: np.ndarray, idx: Sequence[int]) -> Tuple[float, float, float]:
    """O-information of the subset ``idx``: Omega(X_S) = T(X_S) - D(X_S).

    Args:
        r: (d, d) correlation matrix of the full variable set.
        idx: Indices of the subset S (|S| >= 3 is the regime O-information
            is meant for; smaller subsets are allowed for testing but are
            not informative: Omega is identically 0 for |S| < 3).

    Returns:
        A tuple ``(omega, T, D)``, all in nats.
    """
    t = total_correlation(r, idx)
    d = dual_total_correlation(r, idx)
    return t - d, t, d

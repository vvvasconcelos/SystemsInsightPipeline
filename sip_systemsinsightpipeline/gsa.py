"""Global sensitivity analysis: variance-based (Sobol) and moment-independent (delta, PAWN).

This module is the model-agnostic engine behind :meth:`SDM.run_GSA`. It wraps
`SALib <https://salib.readthedocs.io>`_ to provide two complementary families of sensitivity
measures, each returned as a tidy, sorted ``pandas.DataFrame``:

- **Variance-based (Sobol).** First-order ``S1`` and total-order ``ST`` indices from a
  Saltelli design. These decompose the *variance* of the output. Their estimators are
  differences of variance estimates, so for an unimportant parameter (true index ~ 0) they can
  fall slightly outside the valid ``[0, 1]`` range at finite sample size; we project onto
  ``[0, 1]`` and report **BCa** (bias-corrected, accelerated) bootstrap confidence intervals,
  which are asymmetric and respect the boundary.
- **Moment-independent / distributional.** Borgonovo's ``delta`` and ``PAWN`` measure the shift
  in the *whole output distribution* induced by an input, not just its variance. They are
  non-negative by construction and computed on a *given* ``(X, y)`` sample (no special design),
  so they do not suffer the near-zero negativity of variance decomposition.

Keeping the engine separate from the SDM means it can be tested directly on analytic
reference functions (e.g. Ishigami) independently of any system-dynamics model.

References
----------
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.* Wiley.
- Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and
  their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.
- Owen, A. B. (2013). Better estimation of small Sobol' sensitivity indices. *ACM Trans. on
  Modeling and Computer Simulation*, 23(2). (Behaviour of estimators for near-zero indices.)
- Borgonovo, E. (2007). A new uncertainty importance measure. *Reliability Engineering &
  System Safety*, 92(6), 771-784. (The moment-independent ``delta`` measure.)
- Pianosi, F. & Wagener, T. (2015). A simple and efficient method for global sensitivity
  analysis based on cumulative distribution functions. *Env. Modelling & Software*, 67. (PAWN.)
- Efron, B. (1987). Better bootstrap confidence intervals. *JASA*, 82(397), 171-185. (BCa.)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_salib():
    """Import SALib lazily so the package installs/imports without it present."""
    try:
        from SALib.analyze import sobol as _analyze
        from SALib.sample import sobol as _sample
    except ImportError as exc:  # pragma: no cover - exercised only without SALib
        raise ImportError(
            "Global sensitivity analysis requires SALib. Install it with `pip install SALib`."
        ) from exc
    return _sample, _analyze


def sobol_problem(names: Sequence[str], bounds: Sequence[Tuple[float, float]]) -> Dict:
    """Build a SALib problem definition from parameter names and ``(low, high)`` bounds."""
    names = list(names)
    bounds = [[float(lo), float(hi)] for lo, hi in bounds]
    if len(names) != len(bounds):
        raise ValueError("names and bounds must have the same length.")
    if len(names) == 0:
        raise ValueError("At least one free parameter is required for GSA.")
    for name, (lo, hi) in zip(names, bounds):
        if hi < lo:
            raise ValueError(f"Bound for '{name}' has high < low: ({lo}, {hi}).")
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def sobol_sample(problem: Dict, n: int = 1024, *, second_order: bool = False,
                 seed: Optional[int] = None) -> np.ndarray:
    """Draw a Saltelli sample.

    Returns an array of shape ``(n*(d+2), d)`` (or ``(n*(2d+2), d)`` if ``second_order``),
    where ``d = problem['num_vars']``. ``n`` should be a power of two for the Sobol sequence's
    balance properties; SALib warns otherwise.
    """
    sample, _ = _require_salib()
    return sample.sample(problem, int(n), calc_second_order=bool(second_order), seed=seed)


def _sobol_point(A: np.ndarray, B: np.ndarray, AB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Saltelli (2010) first-order and Jansen total-order estimators.

    Identical formulas to SALib's ``sobol.first_order`` / ``sobol.total_order`` (matched in
    tests), evaluated for all parameters at once. ``A``, ``B`` are ``(N,)``; ``AB`` is ``(N, D)``.
    """
    var_y = np.var(np.concatenate([A, B]))
    if var_y == 0:
        z = np.zeros(AB.shape[1])
        return z, z
    s1 = np.mean(B[:, None] * (AB - A[:, None]), axis=0) / var_y
    st = 0.5 * np.mean((A[:, None] - AB) ** 2, axis=0) / var_y
    return s1, st


def _percentile_bounds(point, boot, conf_level, clip):
    if clip:
        point = np.clip(point, 0.0, 1.0)
        boot = np.clip(boot, 0.0, 1.0)
    lo_pct = 100.0 * (1.0 - conf_level) / 2.0
    hi_pct = 100.0 * (1.0 + conf_level) / 2.0
    return np.percentile(boot, lo_pct, axis=0), np.percentile(boot, hi_pct, axis=0)


def _bca_bounds(point, boot, jack, conf_level, clip):
    """Bias-corrected and accelerated (BCa) bootstrap interval (Efron 1987), per parameter.

    ``point`` is ``(D,)``; ``boot`` is ``(B, D)`` bootstrap replicates; ``jack`` is ``(N, D)``
    leave-one-out (jackknife) replicates used to estimate the acceleration. The bias and skew
    corrections are computed on the natural estimator scale; the resulting interval (and the
    point) are then projected onto ``[0, 1]`` if ``clip`` is set.
    """
    from scipy.stats import norm

    n_boot, d = boot.shape
    lo_a = (1.0 - conf_level) / 2.0
    hi_a = (1.0 + conf_level) / 2.0
    low = np.empty(d)
    high = np.empty(d)
    for j in range(d):
        bj = boot[:, j]
        # bias-correction z0 from the fraction of replicates below the point estimate
        prop = np.mean(bj < point[j])
        prop = min(max(prop, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
        z0 = norm.ppf(prop)
        # acceleration from the jackknife distribution's skewness
        jj = jack[:, j]
        centred = jj.mean() - jj
        denom = 6.0 * (np.sum(centred ** 2) ** 1.5)
        a = (np.sum(centred ** 3) / denom) if denom > 0 else 0.0

        def adjusted(alpha):
            z = norm.ppf(alpha)
            return norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

        a1, a2 = adjusted(lo_a), adjusted(hi_a)
        if not (np.isfinite(a1) and np.isfinite(a2)):  # degenerate -> plain percentile
            a1, a2 = lo_a, hi_a
        low[j] = np.percentile(bj, 100.0 * a1)
        high[j] = np.percentile(bj, 100.0 * a2)
    if clip:
        low = np.clip(low, 0.0, 1.0)
        high = np.clip(high, 0.0, 1.0)
    return low, high


def sobol_analyze(problem: Dict, Y: np.ndarray, *, second_order: bool = False,
                  n_bootstrap: int = 200, conf_level: float = 0.95,
                  seed: Optional[int] = None, clip: bool = True, ci: str = "bca") -> pd.DataFrame:
    """Estimate Sobol indices from model outputs ``Y`` and return a tidy DataFrame.

    Columns (sorted by ``ST`` descending): ``parameter``, ``S1``, ``S1_low``, ``S1_high``,
    ``S1_conf``, ``ST``, ``ST_low``, ``ST_high``, ``ST_conf``.

    - ``S1`` / ``ST`` are the point estimates (Saltelli 2010 / Jansen, identical to SALib).
    - ``*_low`` / ``*_high`` are an **asymmetric** confidence interval at ``conf_level``.
      ``ci="bca"`` (default) is a bias-corrected and accelerated bootstrap interval (Efron 1987),
      which corrects both the bias (the bootstrap median drifting from the estimate) and the
      skew of the sampling distribution near the 0/1 boundary; ``ci="percentile"`` is the plain
      percentile bootstrap. A symmetric ``±`` band is deliberately *not* used because it is
      wrong near the boundary.
    - ``*_conf`` is the symmetric half-width ``(high - low) / 2``, kept for quick reading.

    Sobol indices lie in ``[0, 1]``, but their estimators are differences of variance estimates
    and can fall slightly outside that range for an unimportant parameter (true index ~ 0) at
    finite sample size. With ``clip=True`` (default) the point estimate and the interval are
    **projected onto ``[0, 1]``**, so an uninformative parameter reports ``S1 = 0`` with a
    one-sided interval like ``[0, 0.05]`` ("indistinguishable from zero") rather than a spurious
    negative value. Pass ``clip=False`` for the raw estimates (a convergence diagnostic: large
    negative excursions signal that ``n`` is too small).

    The estimator itself cannot be made strictly non-negative without bias (a Sobol index is a
    *ratio* of variances); for a lower-variance point estimator near zero see Owen (2013), and
    for a fully boundary-respecting Bayesian alternative see Gaussian-process emulation
    (Oakley & O'Hagan 2004) — both noted as future options in the methods documentation.

    If ``second_order`` is True, the pairwise ``S2`` matrix and its bootstrap half-width are
    attached on the returned frame as ``.attrs['S2']`` / ``.attrs['S2_conf']``.
    """
    if ci not in ("bca", "percentile"):
        raise ValueError(f"ci must be 'bca' or 'percentile', got {ci!r}.")
    _, analyze = _require_salib()
    from SALib.analyze import sobol as _sobol

    Y = np.asarray(Y, dtype=float).ravel()
    if not np.all(np.isfinite(Y)):
        raise ValueError("Model outputs contain non-finite values; cannot compute indices.")
    if np.var(Y) <= 0:
        raise ValueError(
            "Outcome has (near-)zero variance across the Sobol design, so sensitivity "
            "indices are undefined. Check that the chosen scenario actually drives the "
            "variable of interest and that not all parameters are fixed."
        )

    d = int(problem["num_vars"])
    step = 2 * d + 2 if second_order else d + 2
    n = Y.size // step
    A, B, AB, _ = _sobol.separate_output_values(Y, d, n, second_order)

    s1_pt, st_pt = _sobol_point(A, B, AB)

    rng = np.random.default_rng(seed)
    boot_s1 = np.empty((n_bootstrap, d))
    boot_st = np.empty((n_bootstrap, d))
    for b in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        boot_s1[b], boot_st[b] = _sobol_point(A[idx], B[idx], AB[idx])

    if ci == "bca":
        jack_s1 = np.empty((n, d))
        jack_st = np.empty((n, d))
        all_idx = np.arange(n)
        for i in range(n):
            keep = all_idx != i
            jack_s1[i], jack_st[i] = _sobol_point(A[keep], B[keep], AB[keep])
        s1_low, s1_high = _bca_bounds(s1_pt, boot_s1, jack_s1, conf_level, clip)
        st_low, st_high = _bca_bounds(st_pt, boot_st, jack_st, conf_level, clip)
    else:
        s1_low, s1_high = _percentile_bounds(s1_pt, boot_s1, conf_level, clip)
        st_low, st_high = _percentile_bounds(st_pt, boot_st, conf_level, clip)

    if clip:
        s1_pt = np.clip(s1_pt, 0.0, 1.0)
        st_pt = np.clip(st_pt, 0.0, 1.0)

    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1": s1_pt, "S1_low": s1_low, "S1_high": s1_high, "S1_conf": (s1_high - s1_low) / 2.0,
        "ST": st_pt, "ST_low": st_low, "ST_high": st_high, "ST_conf": (st_high - st_low) / 2.0,
    }).sort_values("ST", ascending=False, ignore_index=True)

    if second_order:
        names = problem["names"]
        Si = analyze.analyze(problem, Y, calc_second_order=True, num_resamples=int(n_bootstrap),
                             conf_level=float(conf_level), seed=seed, print_to_console=False)
        s2 = np.asarray(Si["S2"], dtype=float)
        if clip:
            s2 = np.clip(s2, 0.0, 1.0)
        df.attrs["S2"] = pd.DataFrame(s2, index=names, columns=names)
        df.attrs["S2_conf"] = pd.DataFrame(np.asarray(Si["S2_conf"], dtype=float), index=names, columns=names)
    return df


def run_sobol(names: Sequence[str], bounds: Sequence[Tuple[float, float]],
              evaluate: Callable[[np.ndarray], float], *, n: int = 1024,
              second_order: bool = False, n_bootstrap: int = 200,
              conf_level: float = 0.95, seed: Optional[int] = None, clip: bool = True,
              ci: str = "bca",
              progress: Optional[Callable[[int, int], None]] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """End-to-end Sobol analysis: sample -> evaluate -> analyze.

    ``evaluate`` maps one design row (a 1D array of length ``d``) to a scalar outcome.
    Returns ``(indices_df, design, outputs)`` so the caller can cache the raw ensemble.
    """
    problem = sobol_problem(names, bounds)
    design = sobol_sample(problem, n, second_order=second_order, seed=seed)
    outputs = np.empty(design.shape[0], dtype=float)
    total = design.shape[0]
    for i, row in enumerate(design):
        outputs[i] = float(evaluate(row))
        if progress is not None:
            progress(i + 1, total)
    indices = sobol_analyze(problem, outputs, second_order=second_order,
                            n_bootstrap=n_bootstrap, conf_level=conf_level, seed=seed,
                            clip=clip, ci=ci)
    return indices, design, outputs


def moment_independent(X, y, names: Sequence[str], *, method: str = "delta",
                       bounds: Optional[Sequence[Tuple[float, float]]] = None,
                       n_bootstrap: int = 200, conf_level: float = 0.95,
                       seed: Optional[int] = None) -> pd.DataFrame:
    """Moment-independent (distributional) sensitivity from a *given* ``(X, y)`` sample.

    Unlike variance-based indices, these measure the shift in the **whole output distribution**
    induced by each input, are **non-negative by construction**, and need no special sampling
    design — they run directly on a Monte-Carlo ensemble (e.g. from :meth:`SDM.sample_outcomes`).

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Input ensemble (a DataFrame's column order is used for ``names`` if given).
    y : array-like, shape (n_samples,)
        Scalar outcome for each sample.
    names : sequence of str
        Parameter names (length ``d``).
    method : {"delta", "pawn"}
        ``"delta"`` — Borgonovo's moment-independent ``delta`` measure (the expected
        normalised shift in the output density when an input is fixed); also returns the
        given-data first-order Sobol ``S1`` as a by-product. ``"pawn"`` — the CDF-based PAWN
        measure (Kolmogorov-Smirnov distance between unconditional and conditional output
        distributions), summarised by its median/mean across conditioning slices.
    bounds : optional
        ``(low, high)`` per parameter for the SALib problem; inferred from ``X``'s min/max if
        omitted.

    Returns
    -------
    pandas.DataFrame
        ``"delta"``: ``parameter, delta, delta_conf, S1, S1_conf`` sorted by ``delta`` desc.
        ``"pawn"``: ``parameter, pawn_median, pawn_mean, pawn_cv`` sorted by ``pawn_median`` desc.

    Notes
    -----
    These measures have a small *positive* bias near zero (finite-sample density/CDF estimation),
    so an irrelevant input reads as a small positive value rather than dipping negative — which
    is exactly why they avoid the boundary pathology of variance decomposition.

    References: Borgonovo (2007); Plischke, Borgonovo & Smith (2013) "Global sensitivity
    measures from given data"; Pianosi & Wagener (2015) for PAWN.
    """
    names = list(names)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X must be 2-D with one column per name.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if np.var(y) <= 0:
        raise ValueError("Outcome has (near-)zero variance; sensitivity is undefined.")

    if bounds is None:
        bounds = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(len(names))]
    problem = sobol_problem(names, bounds)

    if method == "delta":
        from SALib.analyze import delta as _delta
        Si = _delta.analyze(problem, X, y, num_resamples=int(n_bootstrap),
                            conf_level=float(conf_level), seed=seed, print_to_console=False)
        return pd.DataFrame({
            "parameter": names,
            "delta": np.asarray(Si["delta"], dtype=float),
            "delta_conf": np.asarray(Si["delta_conf"], dtype=float),
            "S1": np.asarray(Si["S1"], dtype=float),
            "S1_conf": np.asarray(Si["S1_conf"], dtype=float),
        }).sort_values("delta", ascending=False, ignore_index=True)

    if method == "pawn":
        from SALib.analyze import pawn as _pawn
        Si = _pawn.analyze(problem, X, y, seed=seed, print_to_console=False)
        return pd.DataFrame({
            "parameter": names,
            "pawn_median": np.asarray(Si["median"], dtype=float),
            "pawn_mean": np.asarray(Si["mean"], dtype=float),
            "pawn_cv": np.asarray(Si["CV"], dtype=float),
        }).sort_values("pawn_median", ascending=False, ignore_index=True)

    raise ValueError(f"method must be 'delta' or 'pawn', got {method!r}.")


# Outcome reducers shared by run_GSA and sample_outcomes -----------------------------------

def make_reducer(outcome: str = "final", reducer: Optional[Callable] = None,
                 voi: Optional[str] = None) -> Callable[[pd.DataFrame], float]:
    """Return a callable ``df_solution -> float`` for a named outcome or a custom reducer.

    ``outcome`` is one of ``"final"`` (value at the last recorded time), ``"mean"``,
    ``"min"`` or ``"max"`` of the variable-of-interest column over the horizon. A custom
    ``reducer`` (callable taking the single-run solution DataFrame) overrides ``outcome``.
    """
    if reducer is not None:
        return reducer
    if voi is None:
        raise ValueError("voi must be given when no custom reducer is supplied.")

    valid = {"final", "mean", "min", "max"}
    if outcome not in valid:
        raise ValueError(f"outcome must be one of {sorted(valid)} or pass a reducer; got {outcome!r}.")

    def _reduce(df: pd.DataFrame) -> float:
        col = df[voi]
        if outcome == "final":
            return float(col.iloc[-1])
        if outcome == "mean":
            return float(col.mean())
        if outcome == "min":
            return float(col.min())
        return float(col.max())

    return _reduce

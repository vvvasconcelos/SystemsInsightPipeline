"""Variance-based global sensitivity analysis (Sobol indices).

This module is the model-agnostic engine behind :meth:`SDM.run_GSA`. It is a thin wrapper
around `SALib <https://salib.readthedocs.io>`_ that

- builds a Sobol "problem" from named parameters and their bounds,
- draws a Saltelli sample over that problem,
- and turns SALib's analysis into a tidy, sorted ``pandas.DataFrame``.

Keeping the engine separate from the SDM means it can be tested directly on analytic
reference functions (e.g. Ishigami) independently of any system-dynamics model.

References
----------
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.* Wiley.
- Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and
  their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3).
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


def sobol_analyze(problem: Dict, Y: np.ndarray, *, second_order: bool = False,
                  n_bootstrap: int = 200, conf_level: float = 0.95,
                  seed: Optional[int] = None, clip: bool = True) -> pd.DataFrame:
    """Estimate Sobol indices from model outputs ``Y`` and return a tidy DataFrame.

    Columns (sorted by ``ST`` descending): ``parameter``, ``S1``, ``S1_low``, ``S1_high``,
    ``S1_conf``, ``ST``, ``ST_low``, ``ST_high``, ``ST_conf``.

    - ``S1`` / ``ST`` are the point estimates.
    - ``*_low`` / ``*_high`` are an **asymmetric percentile bootstrap** confidence interval at
      ``conf_level`` (the 100·(1-conf)/2 and 100·(1+conf)/2 percentiles of the bootstrap
      replicates). This is more faithful than a symmetric ``±`` band because the sampling
      distribution of a Sobol index is skewed near the 0/1 boundaries.
    - ``*_conf`` is the symmetric half-width ``(high - low) / 2``, kept for quick reading and
      backward compatibility.

    Sobol indices are mathematically bounded to ``[0, 1]``, but their unbiased estimators are
    differences of variance estimates and can fall slightly outside that range for an
    unimportant parameter (true index ≈ 0) at finite sample size. With ``clip=True`` (default)
    the point estimate and the bootstrap replicates are **projected onto ``[0, 1]``** before
    summarising, so an uninformative parameter reports ``S1 = 0`` with an interval like
    ``[0, 0.05]`` ("indistinguishable from zero") rather than a spurious confidently-negative
    value. Pass ``clip=False`` for the raw, unconstrained estimates (useful as a convergence
    diagnostic: large negative excursions signal that ``n`` is too small).

    If ``second_order`` is True, the pairwise ``S2`` matrix and its bootstrap half-width are
    attached on the returned frame as ``.attrs['S2']`` / ``.attrs['S2_conf']``.
    """
    _, analyze = _require_salib()
    Y = np.asarray(Y, dtype=float).ravel()
    if not np.all(np.isfinite(Y)):
        raise ValueError("Model outputs contain non-finite values; cannot compute indices.")
    if np.var(Y) <= 0:
        raise ValueError(
            "Outcome has (near-)zero variance across the Sobol design, so sensitivity "
            "indices are undefined. Check that the chosen scenario actually drives the "
            "variable of interest and that not all parameters are fixed."
        )

    Si = analyze.analyze(problem, Y, calc_second_order=bool(second_order),
                         num_resamples=int(n_bootstrap), conf_level=float(conf_level),
                         seed=seed, keep_resamples=True, print_to_console=False)

    lo_pct = 100.0 * (1.0 - conf_level) / 2.0
    hi_pct = 100.0 * (1.0 + conf_level) / 2.0

    def summarise(point_key, resample_key, z_conf_key):
        point = np.asarray(Si[point_key], dtype=float)
        reps = Si.get(resample_key)
        if reps is not None:
            reps = np.asarray(reps, dtype=float)            # shape (n_bootstrap, d)
            if clip:
                point = np.clip(point, 0.0, 1.0)
                reps = np.clip(reps, 0.0, 1.0)
            low = np.percentile(reps, lo_pct, axis=0)
            high = np.percentile(reps, hi_pct, axis=0)
        else:  # SALib too old to return resamples: fall back to the symmetric band
            half = np.asarray(Si[z_conf_key], dtype=float)
            if clip:
                point = np.clip(point, 0.0, 1.0)
            low, high = point - half, point + half
            if clip:
                low, high = np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)
        return point, low, high, (high - low) / 2.0

    s1, s1_low, s1_high, s1_conf = summarise("S1", "S1_conf_all", "S1_conf")
    st, st_low, st_high, st_conf = summarise("ST", "ST_conf_all", "ST_conf")

    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1": s1, "S1_low": s1_low, "S1_high": s1_high, "S1_conf": s1_conf,
        "ST": st, "ST_low": st_low, "ST_high": st_high, "ST_conf": st_conf,
    }).sort_values("ST", ascending=False, ignore_index=True)

    if second_order:
        names = problem["names"]
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
                            n_bootstrap=n_bootstrap, conf_level=conf_level, seed=seed, clip=clip)
    return indices, design, outputs


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

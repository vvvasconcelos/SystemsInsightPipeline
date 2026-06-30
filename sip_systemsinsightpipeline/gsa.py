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
                  seed: Optional[int] = None) -> pd.DataFrame:
    """Estimate Sobol indices from model outputs ``Y`` and return a tidy DataFrame.

    Columns: ``parameter, S1, S1_conf, ST, ST_conf`` (sorted by ``ST`` descending). The
    ``*_conf`` columns are the half-width of the bootstrap confidence interval at
    ``conf_level``. If ``second_order`` is True, the pairwise ``S2`` matrix is attached on
    the returned frame as ``.attrs['S2']`` / ``.attrs['S2_conf']`` (DataFrames indexed by
    parameter name).
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
                         seed=seed, print_to_console=False)

    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1": np.asarray(Si["S1"], dtype=float),
        "S1_conf": np.asarray(Si["S1_conf"], dtype=float),
        "ST": np.asarray(Si["ST"], dtype=float),
        "ST_conf": np.asarray(Si["ST_conf"], dtype=float),
    }).sort_values("ST", ascending=False, ignore_index=True)

    if second_order:
        names = problem["names"]
        df.attrs["S2"] = pd.DataFrame(np.asarray(Si["S2"], dtype=float), index=names, columns=names)
        df.attrs["S2_conf"] = pd.DataFrame(np.asarray(Si["S2_conf"], dtype=float), index=names, columns=names)
    return df


def run_sobol(names: Sequence[str], bounds: Sequence[Tuple[float, float]],
              evaluate: Callable[[np.ndarray], float], *, n: int = 1024,
              second_order: bool = False, n_bootstrap: int = 200,
              conf_level: float = 0.95, seed: Optional[int] = None,
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
                            n_bootstrap=n_bootstrap, conf_level=conf_level, seed=seed)
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

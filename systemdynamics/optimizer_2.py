"""
sdm_optimizer.py

Intervention optimization module for System Dynamics Models (SDM).

Key design choices (per latest spec):
- Single formulation in expenditure space y (one decision variable per intervention channel)
    y_i >= 0,  sum_i y_i <= budget
    intensity_i = y_i / cost_i
- Multi-start constrained local optimization (SLSQP) with quasi-random (Sobol) starts
- "Local" behavior is achieved by setting n_starts=1 (no separate local implementation)
- Costs must be strictly positive (<= 0 raises)
- Robust VOI extraction uses .iloc[-1]
- Near-optimal solutions use (atol, rtol)
- No np.inf returned to optimizers; use a large finite penalty and track failures
- Dedupe equilibria in y-space using tolerance (no rounding-based uniqueness)
- Reproducible sampling via seed parameter
- Parameter sample ↔ optimum mapping supported via returning/storing parameter samples
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import OptimizeResult
from scipy.stats import qmc
from copy import deepcopy

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass
class OptimizationDiagnostics:
    """Lightweight diagnostics to support debugging and performance tuning."""
    algorithm: str
    budget: float
    n_interventions: int
    n_starts_requested: int
    n_starts_used: int
    n_converged: int
    n_valid: int
    n_failed: int
    total_nfev: int
    total_nit: int
    exception_count: int
    first_exception: Optional[str]
    max_abs_effect_observed: Optional[float]
    penalty_value: float
    seed: Optional[int]


@dataclass
class Equilibrium:
    """A (near-)optimal solution candidate."""
    y: np.ndarray
    intensities: np.ndarray
    effect_size: float
    total_cost: float
    fun: float  # objective value (minimized)


class SDMOptimizer:
    """
    Optimizer for System Dynamics Models (SDM).

    This class optimizes intervention expenditures y under a budget constraint:
        y_i >= 0,  sum(y) <= budget
    and converts to intervention intensities:
        intensity_i = y_i / cost_i

    Parameters
    ----------
    sdm : object
        SDM instance. Expected attributes/methods (as used in your existing codebase):
        - intervention_variables: List[str]
        - intervention_strengths: Dict[str, float]
        - stocks: List[str]
        - constants: List[str]
        - stocks_and_auxiliaries: Iterable[str]
        - variable_of_interest: List[str] or similar
        - interaction_terms: bool / list
        - params_to_A_b(params, constants_values) -> (A, b)
        - run_SDM(x0, constants_values, A, b, params) -> pd.DataFrame
        - sample_model_parameters() -> dict
        - N : int (number of parameter samples, optional)
        - _equation_evaluator with has_custom_equation(var) (optional)

    Notes
    -----
    This optimizer currently uses a multi-start SLSQP strategy. The API is structured
    to allow adding additional algorithms later (e.g., SHGO, CMA-ES, etc.) without
    reintroducing a separate "local" implementation.
    """

    def __init__(self, sdm: Any):
        self.sdm = sdm
        self._last_parameter_samples: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------------------
    # Model execution
    # ---------------------------------------------------------------------
    def run_SDM_with_intervention_intensities(
        self,
        intervention_intensities: Union[np.ndarray, List[float]],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Run SDM with intervention intensities applied.

        Interventions are applied as positive magnitudes; direction is determined
        by the sign of self.sdm.intervention_strengths[var] (or sub-vars for "A+B").
        """
        intervention_intensities = np.asarray(intervention_intensities, dtype=float)

        n_vars = len(self.sdm.intervention_variables)
        if intervention_intensities.shape[0] != n_vars:
            raise ValueError(
                f"Length of intervention_intensities ({intervention_intensities.shape[0]}) "
                f"must match number of intervention variables ({n_vars})."
            )

        # Defensive copy to avoid in-place mutation of caller's params
        params = deepcopy(params)

        # Initialize intervention arrays
        x0 = np.zeros(len(self.sdm.stocks), dtype=np.float64)
        constants_values = np.zeros(len(self.sdm.constants), dtype=np.float64)

        # Preserve and restore SDM mutable state to reduce cross-run contamination
        prev_current = getattr(self.sdm, "_current_intervention_intensities", None)
        self.sdm._current_intervention_intensities = {}

        try:
            for var, intensity in zip(self.sdm.intervention_variables, intervention_intensities):
                if intensity == 0.0:
                    continue

                # Single vs compound intervention channel
                if "+" not in var:
                    intervention_sign = (
                        np.sign(self.sdm.intervention_strengths.get(var, 1.0))
                        if self.sdm.intervention_strengths.get(var, 0.0) != 0.0
                        else 1.0
                    )
                    signed_intensity = float(intensity) * float(intervention_sign)

                    if var in self.sdm.stocks:
                        x0[self.sdm.stocks.index(var)] += signed_intensity
                    elif var in self.sdm.constants:
                        constants_values[self.sdm.constants.index(var)] += signed_intensity
                    else:
                        self.sdm._current_intervention_intensities[var] = signed_intensity
                        # Only set intercept if no custom equation
                        if not (
                            getattr(self.sdm, "_equation_evaluator", None)
                            and self.sdm._equation_evaluator.has_custom_equation(var)
                        ):
                            if var not in params:
                                params[var] = {}
                            params[var]["Intercept"] = signed_intensity

                else:
                    parts = [p.strip() for p in var.split("+")]
                    if len(parts) != 2:
                        raise ValueError(
                            f"Compound intervention variable '{var}' must split into exactly two parts."
                        )
                    var_1, var_2 = parts

                    sign_1 = (
                        np.sign(self.sdm.intervention_strengths.get(var_1, 1.0))
                        if self.sdm.intervention_strengths.get(var_1, 0.0) != 0.0
                        else 1.0
                    )
                    sign_2 = (
                        np.sign(self.sdm.intervention_strengths.get(var_2, 1.0))
                        if self.sdm.intervention_strengths.get(var_2, 0.0) != 0.0
                        else 1.0
                    )

                    delta_1 = float(intensity) * float(sign_1)
                    delta_2 = float(intensity) * float(sign_2)

                    if var_1 in self.sdm.stocks:
                        x0[self.sdm.stocks.index(var_1)] += delta_1
                    elif var_1 in self.sdm.constants:
                        constants_values[self.sdm.constants.index(var_1)] += delta_1
                    else:
                        self.sdm._current_intervention_intensities[var_1] = delta_1
                        if not (
                            getattr(self.sdm, "_equation_evaluator", None)
                            and self.sdm._equation_evaluator.has_custom_equation(var_1)
                        ):
                            if var_1 not in params:
                                params[var_1] = {}
                            params[var_1]["Intercept"] = delta_1

                    if var_2 in self.sdm.stocks:
                        x0[self.sdm.stocks.index(var_2)] += delta_2
                    elif var_2 in self.sdm.constants:
                        constants_values[self.sdm.constants.index(var_2)] += delta_2
                    else:
                        self.sdm._current_intervention_intensities[var_2] = delta_2
                        if not (
                            getattr(self.sdm, "_equation_evaluator", None)
                            and self.sdm._equation_evaluator.has_custom_equation(var_2)
                        ):
                            if var_2 not in params:
                                params[var_2] = {}
                            params[var_2]["Intercept"] = delta_2

            # Get system matrices A and b (if applicable)
            if getattr(self.sdm, "interaction_terms", None):
                A = None
                b = None
            else:
                A, b = self.sdm.params_to_A_b(params, constants_values)

            df_sol = self.sdm.run_SDM(x0, constants_values, A, b, params)
            return df_sol

        finally:
            # Restore state (best-effort)
            self.sdm._current_intervention_intensities = prev_current if prev_current is not None else {}

    # ---------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------
    def optimize_intervention_intensities(
        self,
        params: Dict[str, Any],
        costs: Union[np.ndarray, List[float]],
        variable_of_interest: Optional[str] = None,
        *,
        budget: float = 1.0,
        maximize: bool = True,
        # starts / sampling
        n_starts: Optional[int] = 8,
        seed: Optional[int] = None,
        sobol_power_of_two: bool = True,
        # near-optimal selection (effect size primary, total cost secondary)
        threshold_atol: float = 0.01,
        threshold_rtol: float = 0.0,
        cost_atol: float = 0.01,
        # dedupe in y-space
        y_dedupe_tol: float = 1e-6,
        # optimizer options
        slsqp_maxiter: int = 1000,
        slsqp_ftol: float = 1e-9,
        # penalty behavior
        penalty_value: float = 1e30,
        constraint_tol: float = 1e-12,
        debug: bool = False,
        # extensibility
        algorithm: str = "multistart_slsqp",
        # backwards compatibility alias (optional)
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize intervention strategy for one parameter set.

        Decision variable is y (expenditure per intervention channel):
            y_i >= 0, sum(y) <= budget
        Intensities are computed as intensity_i = y_i / cost_i.

        Parameters
        ----------
        params : dict
            Parameter dictionary for the SDM.
        costs : array-like
            Strictly positive costs for each intervention channel.
        variable_of_interest : str, optional
            Variable to optimize at final time.
        budget : float
            Total budget for expenditures y (default 1.0).
        maximize : bool
            Whether to maximize or minimize the VOI.
        n_starts : int
            Number of multi-start initial points (feasible) for SLSQP. If None, defaults to 30.
        seed : int, optional
            Random seed for reproducible scrambled Sobol sampling.
        sobol_power_of_two : bool
            If True, use next power-of-two >= n_starts for Sobol balance properties.
        threshold_atol, threshold_rtol : float
            Near-optimal inclusion rule for effect size (primary criterion):
              maximize:  best - current <= atol + rtol*abs(best)
              minimize:  current - best <= atol + rtol*abs(best)
        cost_atol : float
            Near-optimal inclusion rule for total cost (secondary criterion):
            Among solutions passing effect size filter, keep those within cost_atol of the minimum cost.
        y_dedupe_tol : float
            Dedupe solutions by Euclidean distance in y-space.
        slsqp_maxiter, slsqp_ftol : optimizer tuning parameters.
        penalty_value : float
            Large finite penalty used when model evaluation fails or returns non-finite VOI.
        debug : bool
            If True, re-raise objective exceptions instead of penalizing.
        algorithm : str
            Currently only "multistart_slsqp" is implemented.
        n_samples : int, optional
            Deprecated alias for n_starts.

        Returns
        -------
        dict
            - method: algorithm name
            - success: bool
            - n_equilibria: int
            - equilibria: list of equilibria dicts
            - best_effect_size: float
            - best_intensities: np.ndarray
            - best_y: np.ndarray
            - optimization_result: OptimizeResult (best run)
            - diagnostics: OptimizationDiagnostics
            - per_start_results: list[OptimizeResult] (all starts, for debugging/tuning)
        """
        if n_samples is not None and n_starts is None:
            warnings.warn("n_samples is deprecated; use n_starts instead. Treating n_samples as n_starts.")
            n_starts = int(n_samples)

        if variable_of_interest is None:
            variable_of_interest = self.sdm.variable_of_interest[0]

        # Warning / early exit if VOI not present
        voi_known = variable_of_interest in getattr(self.sdm, "stocks_and_auxiliaries", [])
        if not voi_known:
            warnings.warn(
                f"Variable of interest '{variable_of_interest}' not found in model variables. "
                f"Skipping optimization for this parameter set."
            )
            diag = OptimizationDiagnostics(
                algorithm=algorithm,
                budget=float(budget),
                n_interventions=len(self.sdm.intervention_variables),
                n_starts_requested=int(n_starts or 0),
                n_starts_used=0,
                n_converged=0,
                n_valid=0,
                n_failed=0,
                total_nfev=0,
                total_nit=0,
                exception_count=0,
                first_exception=None,
                max_abs_effect_observed=None,
                penalty_value=float(penalty_value),
                seed=seed,
            )
            return {
                "method": algorithm,
                "success": False,
                "message": "VOI not found",
                "n_equilibria": 0,
                "equilibria": [],
                "best_effect_size": np.nan,
                "best_intensities": None,
                "best_y": None,
                "optimization_result": None,
                "diagnostics": diag,
                "per_start_results": [],
            }

        costs_array = np.asarray(costs, dtype=float)
        n_interventions = len(self.sdm.intervention_variables)

        if costs_array.shape[0] != n_interventions:
            raise ValueError(
                f"Length of costs ({costs_array.shape[0]}) must match number of intervention variables ({n_interventions})."
            )
        if np.any(~np.isfinite(costs_array)):
            raise ValueError("All costs must be finite numbers.")
        if np.any(costs_array <= 0.0):
            raise ValueError("All costs must be strictly positive (> 0).")

        if not np.isfinite(budget) or budget <= 0.0:
            raise ValueError("budget must be a finite positive number.")

        if n_starts is None:
            n_starts = 8
        if n_starts <= 0:
            raise ValueError("n_starts must be a positive integer.")

        if algorithm != "multistart_slsqp":
            raise ValueError(f"Unknown algorithm '{algorithm}'. Only 'multistart_slsqp' is implemented.")

        # --- Generate feasible start points directly in the simplex sum(y) <= budget
        # Use Sobol in dimension (d+1):
        #   - first d uniforms -> Dirichlet(1) via -log(u) normalization (sum=1)
        #   - last uniform -> radial scaling u^(1/d) for uniform volume in simplex
        n_draw = self._next_power_of_two(n_starts) if sobol_power_of_two else n_starts
        sampler = qmc.Sobol(d=n_interventions + 1, scramble=True, seed=seed)
        u = sampler.random(n=n_draw)

        starts_y = self._sobol_to_simplex_y(u, budget=budget)
        # If user requested fewer starts than drawn for Sobol properties, we use all drawn starts.
        # (This keeps Sobol balance and is usually what you want for multistart.)
        n_starts_used = starts_y.shape[0]

        # --- Objective + monitoring
        exception_count = 0
        first_exception: Optional[str] = None
        max_abs_effect: Optional[float] = None

        def objective(y: np.ndarray) -> float:
            nonlocal exception_count, first_exception, max_abs_effect

            y_arr = np.asarray(y, dtype=float)

            # Defensive check: stay in feasible region
            s = float(np.sum(y_arr))
            if s > budget * (1.0 + 1e-10):
                # Prefer a finite penalty (not inf) for optimizer robustness
                violation = max(0.0, s - budget)
                return float(penalty_value) + float(penalty_value) * violation

            # Convert expenditure y -> intensities
            intensities = y_arr / costs_array  # costs are strictly positive by validation

            try:
                df_sol = self.run_SDM_with_intervention_intensities(intensities, params)
                effect = float(df_sol[variable_of_interest].iloc[-1])

                if not np.isfinite(effect):
                    return float(penalty_value)

                if max_abs_effect is None:
                    max_abs_effect = abs(effect)
                else:
                    max_abs_effect = max(max_abs_effect, abs(effect))

                # Minimize objective
                return -effect if maximize else effect

            except Exception as e:
                exception_count += 1
                if first_exception is None:
                    first_exception = f"{type(e).__name__}: {e}"

                if debug:
                    raise

                return float(penalty_value)

        # Constraint: sum(y) <= budget
        budget_constraint = {"type": "ineq", "fun": lambda y: budget - np.sum(y)}

        # Bounds: each y_i in [0, budget]
        y_bounds = [(0.0, float(budget)) for _ in range(n_interventions)]

        # --- Run local optimization from each start
        per_start_results: List[OptimizeResult] = []
        for y0 in starts_y:
            res = scipy.optimize.minimize(
                objective,
                y0,
                method="SLSQP",
                bounds=y_bounds,
                constraints=budget_constraint,
                options={"maxiter": int(slsqp_maxiter), "ftol": float(slsqp_ftol)},
            )
            per_start_results.append(res)

        # --- Collect valid results
        # Prefer success+finite, else allow finite-only to avoid total failure
        success_finite = [r for r in per_start_results if bool(r.success) and np.isfinite(r.fun)]
        finite_only = [r for r in per_start_results if np.isfinite(r.fun)]
        valid_results = success_finite if len(success_finite) > 0 else finite_only

        if len(valid_results) == 0:
            diag = OptimizationDiagnostics(
                algorithm=algorithm,
                budget=float(budget),
                n_interventions=n_interventions,
                n_starts_requested=int(n_starts),
                n_starts_used=int(n_starts_used),
                n_converged=0,
                n_valid=0,
                n_failed=int(n_starts_used),
                total_nfev=int(sum(getattr(r, "nfev", 0) for r in per_start_results)),
                total_nit=int(sum(getattr(r, "nit", 0) for r in per_start_results)),
                exception_count=int(exception_count),
                first_exception=first_exception,
                max_abs_effect_observed=max_abs_effect,
                penalty_value=float(penalty_value),
                seed=seed,
            )
            return {
                "method": algorithm,
                "success": False,
                "message": "No valid solutions found",
                "n_equilibria": 0,
                "equilibria": [],
                "best_effect_size": np.nan,
                "best_intensities": None,
                "best_y": None,
                "optimization_result": None,
                "diagnostics": diag,
                "per_start_results": per_start_results,
            }

        # Best result (min objective)
        best_res = min(valid_results, key=lambda r: float(r.fun))
        best_fun = float(best_res.fun)
        best_effect = (-best_fun) if maximize else best_fun

        # Warn if penalty might be too small relative to observed effect scales
        # (user asked to flag this possibility)
        if max_abs_effect is not None and max_abs_effect > 0.1 * penalty_value:
            warnings.warn(
                f"Observed |VOI| up to {max_abs_effect:.3g}, which is close to penalty_value={penalty_value:.3g}. "
                "Consider increasing penalty_value to avoid failed evaluations appearing artificially attractive."
            )

        # Near-optimal selection threshold for effect size (primary criterion)
        def near_optimal_effect(effect: float) -> bool:
            tol = float(threshold_atol) + float(threshold_rtol) * abs(float(best_effect))
            if maximize:
                return (best_effect - effect) <= tol
            return (effect - best_effect) <= tol

        # Build equilibria candidates from valid minimizers
        candidates: List[Equilibrium] = []
        for r in valid_results:
            y = np.asarray(r.x, dtype=float)
            y = np.clip(y, 0.0, float(budget))
            # enforce sum constraint gently (numerical)
            s = float(np.sum(y))
            if s > budget * (1.0 + constraint_tol):
                # project down to budget (simple scaling) to keep feasibility
                y = y * (float(budget) / s)

            intensities = y / costs_array
            fun = float(r.fun)
            effect = (-fun) if maximize else fun
            if not np.isfinite(effect):
                continue

            # Primary filter: effect size within tolerance of best
            if near_optimal_effect(effect):
                candidates.append(
                    Equilibrium(
                        y=y,
                        intensities=intensities,
                        effect_size=float(effect),
                        total_cost=float(np.sum(y)),
                        fun=float(fun),
                    )
                )

        # Secondary filter: total cost within tolerance of minimum cost among candidates
        if len(candidates) > 0:
            min_cost = min(c.total_cost for c in candidates)
            candidates = [c for c in candidates if (c.total_cost - min_cost) <= cost_atol]

        # Order candidates by effect size (best first), then by cost (lowest first)
        candidates.sort(key=lambda e: (-e.effect_size if maximize else e.effect_size, e.total_cost))

        # Dedupe by y-space distance
        equilibria: List[Equilibrium] = []
        for cand in candidates:
            if self._is_new_equilibrium(cand.y, [e.y for e in equilibria], tol=y_dedupe_tol):
                equilibria.append(cand)

        # Prepare output equilibria dicts (full precision; no rounding)
        equilibria_out: List[Dict[str, Any]] = []
        for eq in equilibria:
            intervention_dict = {
                name: float(val)
                for name, val in zip(self.sdm.intervention_variables, eq.intensities)
            }
            equilibria_out.append(
                {
                    "interventions": intervention_dict,
                    "intensities": eq.intensities.copy(),
                    "y": eq.y.copy(),
                    "effect_size": float(eq.effect_size),
                    "total_cost": float(eq.total_cost),
                    "fun": float(eq.fun),
                }
            )

        diag = OptimizationDiagnostics(
            algorithm=algorithm,
            budget=float(budget),
            n_interventions=n_interventions,
            n_starts_requested=int(n_starts),
            n_starts_used=int(n_starts_used),
            n_converged=int(sum(1 for r in per_start_results if bool(r.success))),
            n_valid=int(len(valid_results)),
            n_failed=int(n_starts_used - sum(1 for r in per_start_results if bool(r.success))),
            total_nfev=int(sum(getattr(r, "nfev", 0) for r in per_start_results)),
            total_nit=int(sum(getattr(r, "nit", 0) for r in per_start_results)),
            exception_count=int(exception_count),
            first_exception=first_exception,
            max_abs_effect_observed=max_abs_effect,
            penalty_value=float(penalty_value),
            seed=seed,
        )

        return {
            "method": algorithm,
            "success": True,
            "n_equilibria": len(equilibria_out),
            "equilibria": equilibria_out,
            "best_effect_size": float(best_effect),
            "best_intensities": (np.asarray(best_res.x, dtype=float) / costs_array),
            "best_y": np.asarray(best_res.x, dtype=float),
            "optimization_result": best_res,
            "diagnostics": diag,
            "per_start_results": per_start_results,
        }

    # ---------------------------------------------------------------------
    # Parameter-sweep optimization
    # ---------------------------------------------------------------------
    def optimize_across_parameter_samples(
        self,
        costs: Union[np.ndarray, List[float]],
        variable_of_interest: Optional[str] = None,
        *,
        n_parameter_samples: Optional[int] = None,
        # optimization parameters forwarded
        budget: float = 1.0,
        maximize: bool = True,
        n_starts: Optional[int] = None,
        seed: Optional[int] = None,
        sobol_power_of_two: bool = True,
        threshold_atol: float = 0.01,
        threshold_rtol: float = 0.0,
        y_dedupe_tol: float = 1e-6,
        slsqp_maxiter: int = 50,
        slsqp_ftol: float = 1e-3,
        penalty_value: float = 1e30,
        debug: bool = False,
        algorithm: str = "multistart_slsqp",
        # mapping / output control
        store_parameter_samples: bool = True,
        return_parameter_samples: bool = False,
        show_progress: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        Optimize across multiple parameter samples, returning a tidy DataFrame.

        The mapping between sample_idx and parameter set can be preserved by:
        - setting store_parameter_samples=True (default): stored in self._last_parameter_samples
        - setting return_parameter_samples=True: returned alongside the DataFrame

        Output DataFrame columns:
        - sample_idx
        - equilibrium_idx
        - intensity_{intervention_var} (full precision)
        - voi_effect_size
        - total_cost
        - n_equilibria
        - success
        """
        if variable_of_interest is None:
            variable_of_interest = self.sdm.variable_of_interest[0]

        if n_parameter_samples is None:
            n_parameter_samples = int(getattr(self.sdm, "N", 1))

        # Pre-sample parameters to ensure stable mapping sample_idx <-> params
        params_list: List[Dict[str, Any]] = [self.sdm.sample_model_parameters() for _ in range(n_parameter_samples)]

        if store_parameter_samples:
            self._last_parameter_samples = params_list

        iterator: Iterable[int]
        if show_progress and tqdm is not None:
            iterator = tqdm(range(n_parameter_samples), desc=f"Optimizing across parameter samples ({algorithm})")
        else:
            iterator = range(n_parameter_samples)

        rows: List[Dict[str, Any]] = []
        for sample_idx in iterator:
            params = params_list[sample_idx]

            opt = self.optimize_intervention_intensities(
                params=params,
                costs=costs,
                variable_of_interest=variable_of_interest,
                budget=budget,
                maximize=maximize,
                n_starts=n_starts,
                seed=seed,
                sobol_power_of_two=sobol_power_of_two,
                threshold_atol=threshold_atol,
                threshold_rtol=threshold_rtol,
                y_dedupe_tol=y_dedupe_tol,
                slsqp_maxiter=slsqp_maxiter,
                slsqp_ftol=slsqp_ftol,
                penalty_value=penalty_value,
                debug=debug,
                algorithm=algorithm,
            )

            if opt.get("success", False):
                equilibria = opt["equilibria"]
                n_eq = int(opt["n_equilibria"])
                for eq_idx, eq in enumerate(equilibria):
                    row: Dict[str, Any] = {
                        "sample_idx": int(sample_idx),
                        "equilibrium_idx": int(eq_idx),
                        "voi_effect_size": float(eq["effect_size"]),
                        "total_cost": float(eq["total_cost"]),
                        "n_equilibria": int(n_eq),
                        "success": True,
                    }
                    # Flatten nested parameter dict: param_{var}__{param}
                    for var, param_dict in params.items():
                        for pname, pval in param_dict.items():
                            # Handle custom equation parameters (dict or array)
                            if pname.startswith('__eq_params_') and isinstance(pval, dict):
                                for eqk, eqv in pval.items():
                                    row[f"param_{var}__{pname}__{eqk}"] = eqv
                            elif pname.startswith('__eq_params_') and hasattr(pval, '__iter__') and not isinstance(pval, str):
                                for i, eqv in enumerate(pval):
                                    row[f"param_{var}__{pname}__{i}"] = eqv
                            else:
                                row[f"param_{var}__{pname}"] = pval
                    # full precision intensities
                    for name, val in eq["interventions"].items():
                        row[f"intensity_{name}"] = float(val)
                    rows.append(row)
            else:
                row = {
                    "sample_idx": int(sample_idx),
                    "equilibrium_idx": -1,
                    "voi_effect_size": np.nan,
                    "total_cost": np.nan,
                    "n_equilibria": 0,
                    "success": False,
                }
                # Flatten nested parameter dict: param_{var}__{param}
                for var, param_dict in params.items():
                    for pname, pval in param_dict.items():
                        # Handle custom equation parameters (dict or array)
                        if pname.startswith('__eq_params_') and isinstance(pval, dict):
                            for eqk, eqv in pval.items():
                                row[f"param_{var}__{pname}__{eqk}"] = eqv
                        elif pname.startswith('__eq_params_') and hasattr(pval, '__iter__') and not isinstance(pval, str):
                            for i, eqv in enumerate(pval):
                                row[f"param_{var}__{pname}__{i}"] = eqv
                        else:
                            row[f"param_{var}__{pname}"] = pval
                for name in self.sdm.intervention_variables:
                    row[f"intensity_{name}"] = np.nan
                rows.append(row)

        df = pd.DataFrame(rows)

        if return_parameter_samples:
            return df, params_list
        return df

    def get_last_parameter_samples(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the last set of parameter samples used by optimize_across_parameter_samples()."""
        return self._last_parameter_samples

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _next_power_of_two(k: int) -> int:
        if k <= 1:
            return 1
        return 1 << (k - 1).bit_length()

    @staticmethod
    def _sobol_to_simplex_y(u: np.ndarray, *, budget: float) -> np.ndarray:
        """
        Map Sobol points u in [0,1]^(d+1) to feasible y in simplex sum(y) <= budget.

        Construction:
        - Dirichlet(1,...,1) via normalized -log(u_dir) (sum=1)
        - Radial scaling r = u_rad^(1/d) for uniform volume in simplex
        """
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] < 2:
            raise ValueError("u must be 2D with at least 2 columns (d+1).")

        d_plus_1 = u.shape[1]
        d = d_plus_1 - 1
        u_dir = u[:, :d]
        u_rad = u[:, d]

        # avoid log(0)
        eps = np.finfo(float).tiny
        u_dir = np.clip(u_dir, eps, 1.0)
        u_rad = np.clip(u_rad, eps, 1.0)

        z = -np.log(u_dir)
        z_sum = np.sum(z, axis=1, keepdims=True)
        # Dirichlet (uniform on simplex sum=1)
        dirichlet = z / z_sum

        # Uniform-in-volume scaling to sum<=budget simplex
        scale = (u_rad ** (1.0 / d)) * float(budget)
        y = dirichlet * scale[:, None]
        return y

    @staticmethod
    def _is_new_equilibrium(y: np.ndarray, ys_kept: List[np.ndarray], *, tol: float) -> bool:
        if len(ys_kept) == 0:
            return True
        y = np.asarray(y, dtype=float)
        for yk in ys_kept:
            if np.linalg.norm(y - np.asarray(yk, dtype=float)) <= tol:
                return False
        return True
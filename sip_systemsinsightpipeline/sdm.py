import contextlib
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy
import scipy.linalg
import scipy.stats
from copy import deepcopy
from tqdm import tqdm
from tabulate import tabulate


class IntegrationError(RuntimeError):
    """The ODE integration failed, or used more right-hand-side evaluations than allowed."""


#: Default integration settings (copied to ``SDM.solver_options``):
#:
#: - ``rtol``, ``atol``: tolerances of the adaptive ``scipy.integrate.solve_ivp`` methods;
#: - ``max_step``: largest step of the adaptive methods (``inf`` = no limit);
#: - ``dt``: step of the fixed-step methods ``"RK4"``/``"Euler"`` (``None`` = horizon / 1000);
#: - ``max_rhs_evals``: budget of right-hand-side evaluations per simulation for the adaptive
#:   methods. Exceeding it raises :class:`IntegrationError` instead of hanging (discontinuous
#:   equations can make the step size collapse). ``None`` disables the guard.
DEFAULT_SOLVER_OPTIONS = {
    "rtol": 1e-6,
    "atol": 1e-6,
    "max_step": np.inf,
    "dt": None,
    "max_rhs_evals": 200_000,
}
FIXED_STEP_METHODS = ("RK4", "EULER")


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_HELPER_FILES = {os.path.abspath(contextlib.__file__)}


def _user_stacklevel():
    """``stacklevel`` for ``warnings.warn`` that points at the first frame outside SIP."""
    frame, level = sys._getframe(2), 2   # frame 1 is the function calling warnings.warn
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not (filename.startswith(_PACKAGE_DIR) or filename in _HELPER_FILES):
            break
        frame, level = frame.f_back, level + 1
    return level


def magnitude_watch(sdm):
    """Context manager that collects the large-value notice of ``sdm`` and reports it once.

    Works with any object; it is a no-op for objects that are not an :class:`SDM`.
    """
    watch = getattr(sdm, "_magnitude_watch", None)
    return watch() if watch is not None else contextlib.nullcontext()


class SDM:
    def __init__(self, s):
        self.s = s  # keep a reference to the settings (used by plotting helpers)
        self.df_adj = s.df_adj
        self.df_adj_incl_interactions = s.df_adj_incl_interactions
        self.N = s.N
        self.interactions_matrix = s.interactions_matrix
        self.interaction_terms = s.interaction_terms
        self.double_factor_interventions = s.double_factor_interventions
        self.solve_analytically = s.solve_analytically
        self.stocks_and_auxiliaries = s.stocks_and_auxiliaries
        self.stocks_and_constants = s.stocks_and_constants
        self.constants = s.constants
        self.auxiliaries = s.auxiliaries
        self.variables = s.variables
        self.stocks = s.stocks
        self.parameter_value_stocks = s.parameter_value_stocks
        self.parameter_value_aux = s.parameter_value_aux
        self.variable_of_interest = s.variable_of_interest
        self.intervention_variables = s.intervention_variables
        self.prior = "uniform"
        self.intervention_strengths = s.intervention_strengths
        self.auxiliaries_sorted = []  # To store the sorted auxiliaries based on dependencies
        np.random.seed(s.seed)  # Set seed for reproducibility

        # Track current intervention intensities during simulation
        self._current_intervention_intensities = {}  # var -> intensity

        # Custom equations support
        self.equations = getattr(s, 'equations', {})
        self.equation_warnings = getattr(s, 'equation_warnings', [])
        self._equation_evaluator = None
        if self.equations:
            from .equations import EquationEvaluator
            # Use parameter_value_aux as the default range for custom equation parameters
            self._equation_evaluator = EquationEvaluator(
                self.equations,
                parameter_range=(0, s.parameter_value_aux)
            )
            # Print warnings if any
            for warning in self.equation_warnings:
                print(f"Warning: {warning}")

        # Variables whose value (auxiliary) or rate of change (stock) is a custom equation.
        self._eq_vars = {var for var in self.equations
                         if self._equation_evaluator and self._equation_evaluator.has_custom_equation(var)}
        self._eq_param_keys = {var: f'__eq_params_{var}__' for var in self.variables}

        # Models with any valid custom equation must be integrated through the nonlinear
        # ODE path (dz_dt_int -> _compute_equation_value), even with no interaction terms,
        # otherwise the linear A/b shortcut silently ignores the equations.
        self._uses_equations = bool(self._eq_vars)

        # Named parameters (#name): one value per draw, shared by every equation using it.
        # Their ranges come from the workbook's Parameters sheet (s.parameter_priors) and
        # default to [0, parameter_value_aux].
        self.parameter_priors = dict(getattr(s, 'parameter_priors', {}) or {})
        named_users = {}
        for var in self.stocks_and_auxiliaries:
            if var in self._eq_vars:
                for name in self.equations[var].get('named_parameters', []):
                    named_users.setdefault(name, []).append(var)
        self._named_param_users = named_users
        self.named_parameters = sorted(named_users)

        self.num_pars_stocks = int((self.df_adj.loc[self.stocks, :] != 0).sum().sum())
        self.num_pars_auxiliaries = int((self.df_adj.loc[self.auxiliaries, :] != 0).sum().sum())

        stock_indices = [s.variables.index(stock) for stock in self.stocks]
        aux_indices = [s.variables.index(aux) for aux in self.auxiliaries]

        self.num_pars_int_stocks = int(np.count_nonzero(self.interactions_matrix[np.ix_(stock_indices,
                                            np.arange(self.interactions_matrix.shape[1]),
                                            np.arange(self.interactions_matrix.shape[2]))]))

        self.num_pars_int_auxiliaries = int(np.count_nonzero(self.interactions_matrix[np.ix_(aux_indices,
                                            np.arange(self.interactions_matrix.shape[1]),
                                            np.arange(self.interactions_matrix.shape[2]))]))

        # Integration settings. ``solver`` is any scipy.integrate.solve_ivp method ('LSODA'
        # switches automatically between stiff and non-stiff methods since stiffness is not
        # always known) or a fixed-step method ('RK4', 'Euler'); see DEFAULT_SOLVER_OPTIONS.
        self.solver = 'LSODA'
        self.solver_options = dict(DEFAULT_SOLVER_OPTIONS)
        self.t_span = [0.0, s.t_end]
        self.t_eval = np.array(self.t_span)  # Only evaluate final time point by default

        # Notice when simulated stock values become large (|x| > threshold), reported at most
        # once per top-level call. Set to None to disable (e.g. for models in absolute units).
        self.large_value_threshold = 10.0
        self._magnitude_depth = 0
        self._max_abs_stock_value = 0.0

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    # =========================================================================
    # Backward-compatible optimization methods (delegate to SDMOptimizer)
    # For new code, prefer using SDMOptimizer directly:
    #   from sip_systemsinsightpipeline.optimizer import SDMOptimizer
    #   optimizer = SDMOptimizer(sdm)
    # =========================================================================

    def _get_optimizer(self):
        """Lazily create and cache an SDMOptimizer instance."""
        if not hasattr(self, '_optimizer'):
            from .optimizer import SDMOptimizer
            self._optimizer = SDMOptimizer(self)
        return self._optimizer

    def run_SDM_with_intervention_intensities(self, intervention_intensities, params):
        """Run the SDM with intervention intensities. See SDMOptimizer for full documentation."""
        with self._magnitude_watch():
            return self._get_optimizer().run_SDM_with_intervention_intensities(intervention_intensities, params)

    def optimize_intervention_intensities(self, params, costs, variable_of_interest=None, **kwargs):
        """Optimize intervention expenditures under a budget. See SDMOptimizer for full documentation."""
        return self._get_optimizer().optimize_intervention_intensities(
            params, costs, variable_of_interest, **kwargs
        )

    def optimize_across_parameter_samples(self, costs, variable_of_interest=None, **kwargs):
        """Optimize across parameter samples. See SDMOptimizer for full documentation."""
        return self._get_optimizer().optimize_across_parameter_samples(
            costs, variable_of_interest, **kwargs
        )

    # =========================================================================
    # Large-value notice (reported once per top-level call)
    # =========================================================================

    @contextlib.contextmanager
    def _magnitude_watch(self):
        if self._magnitude_depth == 0:
            self._max_abs_stock_value = 0.0
        self._magnitude_depth += 1
        try:
            yield
        finally:
            self._magnitude_depth -= 1
            if self._magnitude_depth == 0:
                self._emit_large_value_notice()

    def _note_magnitude(self, solution):
        solution = np.asarray(solution, dtype=float)
        finite = solution[np.isfinite(solution)]
        if finite.size:
            self._max_abs_stock_value = max(self._max_abs_stock_value, float(np.max(np.abs(finite))))
        if self._magnitude_depth == 0:  # a direct run_SDM call is its own top-level call
            self._emit_large_value_notice()

    def _emit_large_value_notice(self):
        largest, self._max_abs_stock_value = self._max_abs_stock_value, 0.0
        threshold = self.large_value_threshold
        if threshold is not None and largest > threshold:
            warnings.warn(
                f"Simulated stock values reached |x| = {largest:.3g}, larger than {threshold:g} "
                "(sdm.large_value_threshold). In normalised models this suggests that the sampled "
                "parameter ranges (parameter_value_stocks, parameter_value_aux) are too large; for "
                "models in absolute units raise the threshold or set it to None.",
                UserWarning, stacklevel=_user_stacklevel())

    @contextlib.contextmanager
    def _preserve_intervention_state(self):
        previous = self._current_intervention_intensities
        try:
            yield
        finally:
            self._current_intervention_intensities = previous if previous is not None else {}

    # =========================================================================
    # Simulation
    # =========================================================================

    def run_simulations(self):
        """Simulate every intervention for N parameter samples.

        For each parameter sample the model is simulated once without any intervention (the
        *baseline*, stored in ``self.df_sol_baseline_per_sample``) and once per intervention
        variable (``self.df_sol_per_sample[n][i]``). Link parameters of variables with a custom
        equation are removed right after sampling, because the equation replaces the linear
        combination of incoming links.

        Returns ``(df_sol_per_sample, param_samples)``.
        """
        df_sol_per_sample = []  # List for storing the solution dataframes
        df_sol_baseline_per_sample = []
        param_samples = {var: {} for var in self.intervention_variables}  # Parameters across samples

        with self._magnitude_watch(), self._preserve_intervention_state():
            for num in tqdm(range(self.N), desc="Running Simulations"):  # Iterate over the number of samples
                df_sol = []

                params_i = self.sample_model_parameters()  # Sample model parameters

                # Remove link parameters for variables with custom equations right after sampling
                if self._equation_evaluator:
                    for var in params_i:
                        if self._equation_evaluator.has_custom_equation(var):
                            eq_keys = [k for k in params_i[var] if k.startswith('__eq_params_') or k == 'Intercept']
                            params_i[var] = {k: params_i[var][k] for k in eq_keys}

                params_filtered = deepcopy(params_i)

                # No-intervention baseline with the same parameters (effects are measured against it)
                df_sol_baseline_per_sample.append(self._run_baseline(params_filtered))

                for i, var in enumerate(self.intervention_variables):
                    # Set the initial condition for the stocks to zero
                    x0 = np.zeros(len(self.stocks), dtype=np.float64)  # By default no intervention on a stock or constant
                    constants_values = np.zeros(len(self.constants), dtype=np.float64)  # By default no intervention on a constant

                    params = deepcopy(params_filtered)  # Use filtered parameters

                    # Reset current intervention intensities
                    self._current_intervention_intensities = {}

                    if '+' not in var:  # Single factor intervention
                        if var in self.stocks:
                            x0[self.stocks.index(var)] += self.intervention_strengths[var]
                        elif var in self.constants:
                            constants_values[self.constants.index(var)] += self.intervention_strengths[var]
                        else:
                            # Track intervention intensity for custom equations
                            self._current_intervention_intensities[var] = self.intervention_strengths[var]
                            # Only set intercept if no custom equation (custom equation handles $ itself)
                            if not (self._equation_evaluator and self._equation_evaluator.has_custom_equation(var)):
                                params[var]["Intercept"] = self.intervention_strengths[var]

                    else:  # Double factor intervention
                        var_1, var_2 = var.split('+')
                        if var_1 in self.stocks:
                            x0[self.stocks.index(var_1)] += (1/2) * self.intervention_strengths[var_1]
                        elif var_1 in self.constants:
                            constants_values[self.constants.index(var_1)] += (1/2) * self.intervention_strengths[var_1]
                        else:
                            self._current_intervention_intensities[var_1] = (1/2) * self.intervention_strengths[var_1]
                            if not (self._equation_evaluator and self._equation_evaluator.has_custom_equation(var_1)):
                                params[var_1]["Intercept"] = (1/2) * self.intervention_strengths[var_1]

                        if var_2 in self.stocks:
                            x0[self.stocks.index(var_2)] += (1/2) * self.intervention_strengths[var_2]
                        elif var_2 in self.constants:
                            constants_values[self.constants.index(var_2)] += (1/2) * self.intervention_strengths[var_2]
                        else:
                            self._current_intervention_intensities[var_2] = (1/2) * self.intervention_strengths[var_2]
                            if not (self._equation_evaluator and self._equation_evaluator.has_custom_equation(var_2)):
                                params[var_2]["Intercept"] = (1/2) * self.intervention_strengths[var_2]

                    if self.interaction_terms or self._uses_equations:
                        A = None
                        b = None
                    else:  # Get the system matrices A and b
                        A, b = self.params_to_A_b(params, constants_values)

                    df_sol_per_intervention = self.run_SDM(x0, constants_values, A, b, params)
                    df_sol += [df_sol_per_intervention]

                    # Store the model parameters
                    if num == 0:
                        param_samples[var] = {target: {source: [params[target][source]] for source in params[target]} for target in params}
                    else:
                        for target in params:
                            for source in params[target]:
                                param_samples[var][target][source] += [params[target][source]]

                df_sol_per_sample += [df_sol]

        self.df_sol_per_sample = df_sol_per_sample
        self.df_sol_baseline_per_sample = df_sol_baseline_per_sample
        self.param_samples = param_samples
        return df_sol_per_sample, param_samples

    def _run_baseline(self, params):
        """Simulate the model without any intervention for the given parameters."""
        x0 = np.zeros(len(self.stocks), dtype=np.float64)
        constants_values = np.zeros(len(self.constants), dtype=np.float64)
        with self._preserve_intervention_state():
            self._current_intervention_intensities = {}
            if self.interaction_terms or self._uses_equations:
                A, b = None, None
            else:
                A, b = self.params_to_A_b(params, constants_values)
            return self.run_SDM(x0, constants_values, A, b, params)

    def run_SDM(self, x0, constants_values, A, b, params):
        """ Run the SDM and return a dataframe with all the variables at every time step, including auxiliaries.
        """
        if self.interaction_terms or self._uses_equations:
            solution = self._integrate(self.dz_dt_int, x0, (constants_values, params))
        else:  # Linear system
            if self.solve_analytically:
                solution = self.analytical_solution(self.t_eval[:, None], x0, A, b)
            else:
                solution = self._integrate(self.solve_sdm_linear, x0, (A, b), jac=self.jac_linear)

        self._note_magnitude(solution)

        df_sol = pd.DataFrame(solution, columns=self.stocks, index=self.t_eval)
        df_sol["Time"] = df_sol.index

        # Add constants to the dataframe
        for i, const in enumerate(self.constants):
            df_sol.loc[:, const] = constants_values[i]

        # Evaluate auxiliaries using the parameter dictionary and add to dataframe
        params_aux_only = {var : params[var] for var in self.auxiliaries}
        df_sol_with_aux = self.evaluate_auxiliaries(params_aux_only, df_sol)
        return df_sol_with_aux

    # -------------------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------------------

    def _solver_settings(self):
        opts = dict(DEFAULT_SOLVER_OPTIONS)
        opts.update(self.solver_options or {})
        unknown = set(opts) - set(DEFAULT_SOLVER_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown solver option(s) {sorted(unknown)}; valid options are "
                             f"{sorted(DEFAULT_SOLVER_OPTIONS)}.")
        return opts

    def _integrate(self, fun, x0, args, jac=None):
        """Integrate ``dx/dt = fun(t, x, *args)`` over ``self.t_span``; return values at ``self.t_eval``.

        Uses ``self.solver`` and ``self.solver_options`` (see ``DEFAULT_SOLVER_OPTIONS``).
        Raises :class:`IntegrationError` if the solver fails or exceeds ``max_rhs_evals``.
        """
        method = str(self.solver)
        opts = self._solver_settings()
        if method.upper() in FIXED_STEP_METHODS:
            return self._integrate_fixed_step(fun, x0, args, method.upper(), opts["dt"])

        t0, t1 = float(self.t_span[0]), float(self.t_span[1])
        budget = opts["max_rhs_evals"]
        state = {"n": 0, "t": t0}

        def rhs(t, y, *a):
            state["n"] += 1
            if t > state["t"]:
                state["t"] = t
            if budget is not None and state["n"] > budget:
                raise IntegrationError(
                    f"The ODE solver ({method}) did not finish: it used more than {int(budget):,} "
                    f"right-hand-side evaluations and stalled near t = {state['t']:.6g} (of {t1:g}). "
                    "This usually means a discontinuous equation (a switch such as (S < 0.5)) makes "
                    "the adaptive step size collapse. Options: a fixed-step method "
                    "(sdm.solver = 'RK4' with sdm.solver_options['dt']), sdm.solver = 'BDF', or a "
                    "larger sdm.solver_options['max_rhs_evals'] (None disables this guard).")
            return fun(t, y, *a)

        kwargs = dict(t_eval=self.t_eval, method=method, rtol=opts["rtol"], atol=opts["atol"], args=args)
        if opts["max_step"] is not None and np.isfinite(opts["max_step"]):
            kwargs["max_step"] = float(opts["max_step"])
        if jac is not None and method in ("LSODA", "BDF", "Radau"):
            kwargs["jac"] = jac
        sol = solve_ivp(rhs, self.t_span, x0, **kwargs)
        if not sol.success or sol.y.shape[1] != len(self.t_eval):
            reached = sol.t[-1] if len(sol.t) else t0
            raise IntegrationError(f"The ODE solver ({method}) failed at t = {reached:.6g} (of {t1:g}): "
                                   f"{sol.message}")
        return sol.y.T

    def _integrate_fixed_step(self, fun, x0, args, method, dt):
        """Classical fixed-step integration ('RK4' or 'EULER').

        Every recorded time in ``self.t_eval`` is hit exactly: each interval between recorded
        times is split into equal sub-steps no longer than ``dt``.
        """
        t_eval = np.asarray(self.t_eval, dtype=float)
        t0, t1 = float(self.t_span[0]), float(self.t_span[1])
        if dt is None:
            dt = (t1 - t0) / 1000.0
        dt = float(dt)
        if not dt > 0:
            raise ValueError("solver_options['dt'] must be positive.")
        if t_eval.size and (t_eval[0] < t0 - 1e-12 or np.any(np.diff(t_eval) < 0)):
            raise ValueError("t_eval must be increasing and start at or after t_span[0] for fixed-step solvers.")

        x = np.array(x0, dtype=float)
        t = t0
        out = np.empty((t_eval.size, x.size), dtype=float)
        for k, t_rec in enumerate(t_eval):
            span = t_rec - t
            if span > 0:
                n_sub = max(1, int(np.ceil(span / dt - 1e-9)))
                h = span / n_sub
                for j in range(n_sub):
                    tj = t + j * h
                    if method == "EULER":
                        x = x + h * np.asarray(fun(tj, x, *args), dtype=float)
                    else:
                        k1 = np.asarray(fun(tj, x, *args), dtype=float)
                        k2 = np.asarray(fun(tj + h / 2, x + (h / 2) * k1, *args), dtype=float)
                        k3 = np.asarray(fun(tj + h / 2, x + (h / 2) * k2, *args), dtype=float)
                        k4 = np.asarray(fun(tj + h, x + h * k3, *args), dtype=float)
                        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                t = t_rec
            out[k] = x
        return out

    def dz_dt_int(self, t, z, constants_values, params):
        """Compute dz/dt for the nonlinear SDM (interaction terms and/or custom equations)."""
        # Compute all variable values (stocks, constants, auxiliaries)
        vals = self._compute_all_auxiliaries(z, constants_values, params, t)

        # Compute stock derivatives using shared equation evaluation
        dz = np.zeros(len(self.stocks))
        for i, stock in enumerate(self.stocks):
            dz[i] = self._compute_equation_value(stock, params, vals, t)
        return dz

    def _compute_equation_value(self, var, params, vals, t=0.0):
        """
        Compute the value of any equation (auxiliary value or stock derivative).

        This is the single source of truth for equation evaluation,
        used by dz_dt_int (during ODE integration) for both auxiliaries and
        stock derivatives, and by evaluate_auxiliaries (for post-simulation output).

        If a custom equation is defined for the variable (from the Equation column
        in the Kumu Excel file), that equation will be used instead of the default
        linear combination.

        Args:
            var: Name of the variable (auxiliary or stock)
            params: Parameter dictionary with coefficients (also contains equation parameters)
            vals: Dictionary of current variable values (stocks, constants, and already-computed auxiliaries)
            t: simulation time (available to custom equations as ``t``)

        Returns:
            float: Computed equation value
        """
        # Check if this variable has a custom equation
        if var in self._eq_vars:
            # Get the sampled parameters for this equation from params[var] dict
            eq_params = params[var].get(self._eq_param_keys[var], {})
            # Get intervention intensity for this variable (default 0.0 if not intervened)
            intervention = self._current_intervention_intensities.get(var, 0.0)
            return self._equation_evaluator.evaluate(var, vals, eq_params, intervention, t)

        # Default: linear combination based on adjacency matrix
        total = 0.0
        for pred, coef in params[var].items():
            # Skip special equation parameter entries
            if pred.startswith('__eq_params_'):
                continue
            if pred == 'Intercept':
                total += coef
            elif '*' in pred:
                # Handle both " * " and "*" separators for robustness
                if ' * ' in pred:
                    v1, v2 = pred.split(' * ')
                else:
                    v1, v2 = [p.strip() for p in pred.split('*')]
                total += coef * vals[v1] * vals[v2]
            else:
                total += coef * vals[pred]
        return total

    def _compute_all_auxiliaries(self, stock_vals, const_vals, params, t=0.0):
        """
        Compute all auxiliary values given stock and constant values.

        Args:
            stock_vals: Array or list of stock values (in order of self.stocks)
            const_vals: Array or list of constant values (in order of self.constants)
            params: Parameter dictionary
            t: simulation time

        Returns:
            dict: All variable values including computed auxiliaries
        """
        # Initialize with stocks and constants
        vals = {var: 0.0 for var in self.variables}
        for s, v in zip(self.stocks, stock_vals):
            vals[s] = v
        for c, v in zip(self.constants, const_vals):
            vals[c] = v

        # Ensure auxiliaries are sorted by dependency order
        if self.auxiliaries_sorted == []:
            self.auxiliaries_sorted = self.sort_auxiliaries(params)

        # Compute auxiliaries in dependency order
        for aux in self.auxiliaries_sorted:
            vals[aux] = self._compute_equation_value(aux, params, vals, t)

        return vals

    def params_to_A_b(self, params, constants_values):
        """
        Construct system matrices A and b from SDM-style parameter dictionary.
        """
        constants_values = np.asarray(constants_values, dtype=np.float64)
        n_stock = len(self.stocks)
        n_aux = len(self.auxiliaries)
        n_const = len(self.constants)

        # Create index mappings based on the list structures
        stock_idx = {s: i for i, s in enumerate(self.stocks)}
        aux_idx = {a: i for i, a in enumerate(self.auxiliaries)}
        const_idx = {c: i for i, c in enumerate(self.constants)}

        param_mat = np.zeros((n_stock + n_aux, n_stock + n_aux + n_const), dtype=np.float64)
        intercept_aux = np.zeros(n_aux, dtype=np.float64)
        intercept_stock = np.zeros(n_stock, dtype=np.float64)

        for target, effects in params.items():
            for source, val in effects.items():
                if source == 'Intercept':
                    if target in self.auxiliaries:
                        intercept_aux[aux_idx[target]] = val
                    elif target in self.stocks:
                        intercept_stock[stock_idx[target]] = val
                else:
                    # Determine target row index
                    if target in self.stocks:
                        target_row = stock_idx[target]
                    elif target in self.auxiliaries:
                        target_row = n_stock + aux_idx[target]
                    else:
                        continue  # Skip if target is not in stocks or auxiliaries

                    # Determine source column index
                    if source in self.stocks:
                        source_col = stock_idx[source]
                    elif source in self.auxiliaries:
                        source_col = n_stock + aux_idx[source]
                    elif source in self.constants:
                        source_col = n_stock + n_aux + const_idx[source]
                    else:
                        continue  # Skip if source is not in stocks, auxiliaries, or constants

                    param_mat[target_row, source_col] = val

        # Extract submatrices using explicit indices
        # Stock equations (rows 0:n_stock)
        stock_to_stocks = param_mat[:n_stock, :n_stock]
        stock_to_aux = param_mat[:n_stock, n_stock:n_stock+n_aux]
        stock_to_const = param_mat[:n_stock, n_stock+n_aux:]

        # Auxiliary equations (rows n_stock:n_stock+n_aux)
        aux_to_stocks = param_mat[n_stock:n_stock+n_aux, :n_stock]
        aux_to_aux = param_mat[n_stock:n_stock+n_aux, n_stock:n_stock+n_aux]
        aux_to_const = param_mat[n_stock:n_stock+n_aux, n_stock+n_aux:]

        # Solve for effective auxiliary influence
        I_aux = np.eye(n_aux)
        aux_system = I_aux - aux_to_aux

        try:
            aux_system_inv = np.linalg.inv(aux_system)
        except np.linalg.LinAlgError:
            raise ValueError("Auxiliary system (I - A_aux_to_aux) is singular. Check for circular dependencies among auxiliaries.")

        # Effective auxiliary response to stocks
        effective_aux_to_stocks = aux_system_inv @ aux_to_stocks

        # Compute A matrix
        A = stock_to_stocks + stock_to_aux @ effective_aux_to_stocks

        # Compute b vector
        if n_const > 0:
            # How constants affect auxiliaries (after resolving aux-aux dependencies)
            effective_aux_const = aux_system_inv @ (aux_to_const @ constants_values)
            # Direct constant effect on stocks
            bias_from_const = stock_to_const @ constants_values
            # Indirect constant effect through auxiliaries
            bias_from_const += stock_to_aux @ effective_aux_const
        else:
            bias_from_const = np.zeros(n_stock, dtype=np.float64)

        # Auxiliary intercept contributions (after resolving aux-aux dependencies)
        effective_aux_intercept = aux_system_inv @ intercept_aux
        bias_from_aux = stock_to_aux @ effective_aux_intercept

        # Total bias vector
        b = intercept_stock + bias_from_aux + bias_from_const

        return A, b

    def sort_auxiliaries(self, params):
        """Order the auxiliaries so that each is computed after the auxiliaries it depends on.

        The dependencies of an auxiliary with a custom equation are the variables the equation
        references; otherwise they are the sources of its linear and interaction terms in
        ``params``. Raises ``ValueError`` naming the auxiliaries involved when they depend on
        each other in a cycle (an algebraic loop).
        """
        aux_set = set(self.auxiliaries)
        remaining = {}
        for a in self.auxiliaries:
            if a in self._eq_vars:
                used = self._equation_evaluator.get_equation_info(a).get('variables_used', set())
                deps = {v for v in used if v in aux_set}
            else:
                deps = set()
                for key in params.get(a, {}):
                    if key == 'Intercept' or str(key).startswith('__eq_params_'):
                        continue
                    parts = [p.strip() for p in key.split('*')] if '*' in key else [key]
                    deps.update(p for p in parts if p in aux_set)
            remaining[a] = deps

        auxiliaries_sorted = []
        while remaining:
            ready = [a for a in self.auxiliaries if a in remaining and not remaining[a]]
            if not ready:
                # follow unresolved dependencies until a variable repeats -> one cycle
                node, path = next(iter(remaining)), []
                while node not in path:
                    path.append(node)
                    node = sorted(remaining[node])[0]
                cycle = path[path.index(node):] + [node]
                raise ValueError(
                    "Circular dependency among auxiliaries (an algebraic loop): "
                    + " -> ".join(cycle) + ". Auxiliaries are computed instantaneously; break the "
                    "loop with a stock, or check that no variable name in an equation is mistyped.")
            auxiliaries_sorted.extend(ready)
            for r in ready:
                remaining.pop(r)
            for deps in remaining.values():
                deps.difference_update(ready)
        return auxiliaries_sorted

    def evaluate_auxiliaries(self, params, df_sol):
        """ Evaluate the auxiliary variables at each time step.
        Input: original parameter dictionary with auxiliary terms, and the solution dataframe
        Output: dataframe with added auxiliary values at each time step

        Uses _compute_equation_value as single source of truth for auxiliary computation.
        """
        # No auxiliaries: nothing to add. (This early exit matters: the per-row pandas
        # access below used to cost ~a third of total simulation time on aux-free models.)
        if not self.auxiliaries:
            return df_sol

        if self.auxiliaries_sorted == []:  # If not already sorted
            self.auxiliaries_sorted = self.sort_auxiliaries(params)

        # Work on plain numpy arrays; per-cell .loc writes are pathologically slow.
        base_cols = [c for c in df_sol.columns if c != 'Time' and c not in self.auxiliaries]
        base = {c: df_sol[c].to_numpy() for c in base_cols}
        times = np.asarray(df_sol.index, dtype=float)
        n_t = len(df_sol.index)
        aux_arrays = {aux: np.empty(n_t, dtype=np.float64) for aux in self.auxiliaries_sorted}

        for k in range(n_t):
            vals = {c: base[c][k] for c in base_cols}
            for aux in self.auxiliaries_sorted:
                v = self._compute_equation_value(aux, params, vals, times[k])
                vals[aux] = v
                aux_arrays[aux][k] = v

        df_sol_with_auxiliaries = df_sol.copy()
        for aux in self.auxiliaries_sorted:
            df_sol_with_auxiliaries[aux] = aux_arrays[aux]
        return df_sol_with_auxiliaries

    # =========================================================================
    # Intervention effects
    # =========================================================================

    def get_intervention_effects(self, relative_to="baseline"):
        """Effect of each intervention on each variable of interest, per parameter sample.

        Requires :meth:`run_simulations`. By default (``relative_to="baseline"``) the effect is
        the difference, at the final time, between the run with the intervention and the
        no-intervention baseline run with the *same* parameter sample:
        ``VOI(intervention, t_end) - VOI(baseline, t_end)``. For models whose zero state is an
        equilibrium (all pure causal-loop-diagram models) the baseline is zero, so this equals
        the final value. ``relative_to=None`` returns the raw final values instead.

        Returns ``{voi: {intervention: [effect per sample]}}``, each inner dict sorted by the
        median absolute effect (largest first).
        """
        if relative_to not in ("baseline", None):
            raise ValueError(f"relative_to must be 'baseline' or None, got {relative_to!r}.")
        if getattr(self, 'df_sol_per_sample', None) is None:
            raise ValueError("Call run_simulations() before get_intervention_effects().")
        baseline = getattr(self, 'df_sol_baseline_per_sample', None)
        if relative_to == "baseline" and baseline is None:
            raise ValueError("No baseline runs found; call run_simulations() again, or pass relative_to=None.")

        t_end = self.t_eval[-1]
        n_samples = len(self.df_sol_per_sample)
        intervention_effects_per_voi = {voi: {} for voi in self.variable_of_interest}
        for voi in self.variable_of_interest:
            intervention_effects = {}
            for i, i_v in enumerate(self.intervention_variables):
                values = []
                for n in range(n_samples):
                    value = self.df_sol_per_sample[n][i].loc[t_end, voi]
                    if relative_to == "baseline":
                        value = value - baseline[n].loc[t_end, voi]
                    values.append(value)
                intervention_effects[i_v] = values
            # Sort the dictionary by the median intervention effect
            intervention_effects = dict(sorted(intervention_effects.items(),
                                               key=lambda item: np.median(np.abs(item[1])), reverse=True))

            self.intervention_effects = intervention_effects
            intervention_effects_per_voi[voi] = intervention_effects

        return intervention_effects_per_voi

    @staticmethod
    def get_top_interventions(intervention_effects, top_plot=None):
        """ Get the names of the top interventions based on median effect. """
        # Convert intervention effects to DataFrame
        df_SA = pd.DataFrame(intervention_effects)

        # Sort interventions by the absolute median of their effects
        sorted_columns = df_SA.abs().median().sort_values(ascending=False).index

        # If top_plot is specified, limit to the top X interventions
        if top_plot is not None:
            sorted_columns = sorted_columns[:top_plot]
        return list(sorted_columns)

    # =========================================================================
    # Parameter sampling
    # =========================================================================

    def named_parameter_bounds(self, name):
        """``(low, high)`` sampling range of the named parameter ``#name``."""
        low, high = self.parameter_priors.get(name, (0.0, self.parameter_value_aux))
        return float(low), float(high)

    def _sample_named_parameters(self):
        values = {}
        for name in self.named_parameters:
            low, high = self.named_parameter_bounds(name)
            values[name] = low if high <= low else float(np.random.uniform(low, high))
        return values

    def sample_model_parameters(self):
        """ Sample from the model parameters using a bounded uniform distribution.
            The possible parameters are given by the adjacency and interactions matrices.

            Local equation parameters (#1, #2, ...) are drawn from [0, parameter_value_aux].
            Named equation parameters (#name) are drawn once per call and shared by all the
            equations using them, from their range in ``parameter_priors`` (the workbook's
            Parameters sheet) or [0, parameter_value_aux]; they are drawn after all other
            parameters, so models without them keep the same random sequence.
        """
        params = {var : {} for var in self.stocks_and_auxiliaries}

        if self.prior == "uniform":
            sample_pars_stocks = np.random.uniform(0, self.parameter_value_stocks, size=(self.num_pars_stocks))
            sample_pars_auxiliaries = np.random.uniform(0, self.parameter_value_aux, size=(self.num_pars_auxiliaries))
            sample_pars_int_stocks = np.random.uniform(0, self.parameter_value_stocks/2, size=(self.num_pars_int_stocks))
            sample_pars_int_auxiliaries = np.random.uniform(0, self.parameter_value_aux/2, size=(self.num_pars_int_auxiliaries))

        par_count_auxiliaries = 0
        par_count_stocks = 0
        par_int_count_stocks = 0
        par_int_count_auxiliaries = 0

        for var in self.stocks_and_auxiliaries:
            # The interactions matrix is indexed by position in self.variables, which differs
            # from the position in stocks_and_auxiliaries once constants interleave with them
            i = self.variables.index(var)
            params[var]["Intercept"] = 0

            # If this variable has a custom equation, add custom equation parameters, but also
            # ensure all incoming links are present for dependency sorting
            if self._equation_evaluator and self._equation_evaluator.has_custom_equation(var):
                eq_params = self._equation_evaluator.sample_equation_parameters(var)
                params[var][f'__eq_params_{var}__'] = eq_params
                # Reintroduce all incoming links from adjacency matrix (set to 0 if not sampled)
                incoming_links = [var_2 for var_2 in self.variables if self.df_adj.loc[var, var_2] != 0]
                for var_2 in incoming_links:
                    if var_2 not in params[var]:
                        params[var][var_2] = 0.0

            # Pairwise interactions
            for j, var_2 in enumerate(self.variables):
                if self.df_adj.loc[var, var_2] != 0:
                    if self.df_adj.loc[var, var_2] == -999:
                        if var in self.stocks:  # If the variable is a stock
                            params[var][var_2] = (sample_pars_stocks[par_count_stocks] * 2) - self.parameter_value_stocks
                            par_count_stocks += 1
                        else:  # If the variable is an auxiliary
                            params[var][var_2] = (sample_pars_auxiliaries[par_count_auxiliaries] * 2) - self.parameter_value_aux
                            par_count_auxiliaries += 1
                    else:
                        if var in self.stocks:
                            params[var][var_2] = self.df_adj.loc[var, var_2] * sample_pars_stocks[par_count_stocks]
                            par_count_stocks += 1
                        else:
                            params[var][var_2] = self.df_adj.loc[var, var_2] * sample_pars_auxiliaries[par_count_auxiliaries]
                            par_count_auxiliaries += 1

                # 2nd-order interaction terms
                if self.interaction_terms:
                    for k, var_3 in enumerate(self.variables):
                        if self.interactions_matrix[i, j, k] != 0:
                            if self.df_adj.loc[var, var_2] == -999:
                                if var in self.stocks:
                                    params[var][var_2 + " * " + var_3] = (sample_pars_int_stocks[par_int_count_stocks] * 2) - self.parameter_value_stocks
                                    par_int_count_stocks += 1
                                elif var in self.auxiliaries:
                                    params[var][var_2 + " * " + var_3] = (sample_pars_int_auxiliaries[par_int_count_auxiliaries] * 2) - self.parameter_value_aux
                                    par_int_count_auxiliaries += 1
                            else:
                                if var in self.stocks:
                                    params[var][var_2 + " * " + var_3] = self.interactions_matrix[i, j, k] * sample_pars_int_stocks[par_int_count_stocks]
                                    par_int_count_stocks += 1
                                if var in self.auxiliaries:
                                    params[var][var_2 + " * " + var_3] = self.interactions_matrix[i, j, k] * sample_pars_int_auxiliaries[par_int_count_auxiliaries]
                                    par_int_count_auxiliaries += 1

        # Named parameters: one draw each, shared by every equation that uses it
        if self.named_parameters:
            for name, value in self._sample_named_parameters().items():
                for var in self._named_param_users[name]:
                    params[var][self._eq_param_keys[var]][f'#{name}'] = value

        self.params = params
        return params

    # =========================================================================
    # Global sensitivity analysis (Sobol) and ensemble sampling
    #
    # These reuse the uniform priors that sample_model_parameters draws from, but
    # map a flat design vector deterministically into the nested params dict so a
    # Saltelli/Sobol design can be evaluated. See gsa.py for the index estimator.
    # =========================================================================

    def _gsa_parameter_spec(self, bounds=None):
        """Enumerate the model's free (uncertain) parameters with their sampling bounds.

        Returns ``(free, fixed)`` lists of entry dicts. Each entry records where the value
        belongs in the nested ``params`` dict (``target`` + ``kind`` + ``key``) and its
        ``(low, high)`` bounds. The bounds mirror exactly what ``sample_model_parameters``
        draws from, so a GSA samples the same prior:

        - link coefficient of a non-equation stock/auxiliary (``"Target <- Source"``): known
          polarity ``+`` -> ``(0, pv)``, ``-`` -> ``(-pv, 0)``, unknown -> ``(-pv, pv)``, with
          ``pv = parameter_value_stocks`` for stocks and ``parameter_value_aux`` for auxiliaries;
        - each local ``#k`` parameter of a custom equation (``"Var | #k"``): ``(0, parameter_value_aux)``;
        - each named parameter (``"#name"``): its range in ``parameter_priors`` (the workbook's
          Parameters sheet), else ``(0, parameter_value_aux)``; one value for all its equations;
        - interaction terms (only when the model has them) mirror the same formulas at half range.

        Equation variables contribute only their ``#`` parameters (their incoming-link
        coefficients are stripped before simulation, so they do not affect the outcome).
        ``bounds`` is an optional ``{name: (low, high)}`` override; a parameter whose final
        bounds satisfy ``low == high`` is treated as fixed and reported separately.
        """
        bounds = bounds or {}
        entries = []

        def add(name, low, high, kind, target, key):
            low, high = bounds.get(name, (low, high))
            entries.append({"name": name, "low": float(low), "high": float(high),
                            "kind": kind, "target": target, "key": key})

        for var in self.stocks_and_auxiliaries:
            is_stock = var in self.stocks
            pv = self.parameter_value_stocks if is_stock else self.parameter_value_aux
            has_eq = bool(self._equation_evaluator) and self._equation_evaluator.has_custom_equation(var)

            if has_eq:
                for idx in self.equations[var]["parameter_indices"]:
                    add(f"{var} | #{idx}", 0.0, self.parameter_value_aux, "eq", var, f"#{idx}")
                continue

            i = self.variables.index(var)
            for j, var_2 in enumerate(self.variables):
                pol = self.df_adj.loc[var, var_2]
                if pol == 0:
                    continue
                if pol == -999:
                    lo, hi = -pv, pv
                elif pol > 0:
                    lo, hi = 0.0, pv
                else:
                    lo, hi = -pv, 0.0
                add(f"{var} <- {var_2}", lo, hi, "link", var, var_2)

                if self.interaction_terms:
                    for k, var_3 in enumerate(self.variables):
                        ival = self.interactions_matrix[i, j, k]
                        if ival == 0:
                            continue
                        if pol == -999:
                            lo_i, hi_i = -pv, 0.0
                        elif ival > 0:
                            lo_i, hi_i = 0.0, pv / 2
                        else:
                            lo_i, hi_i = -pv / 2, 0.0
                        key = var_2 + " * " + var_3
                        add(f"{var} <- {key}", lo_i, hi_i, "interaction", var, key)

        for name in self.named_parameters:
            lo, hi = self.named_parameter_bounds(name)
            add(f"#{name}", lo, hi, "named", None, f"#{name}")

        unknown = set(bounds) - {e["name"] for e in entries}
        if unknown:
            warnings.warn(f"bounds given for unknown parameter(s) {sorted(unknown)}; they are ignored. "
                          "Parameter names look like 'Target <- Source', 'Var | #1' or '#name'.")

        free = [e for e in entries if e["high"] > e["low"]]
        fixed = [e for e in entries if e["high"] <= e["low"]]
        return free, fixed

    def _gsa_params_from_vector(self, vec, free, fixed):
        """Build the nested ``params`` dict from a flat design row (mirrors the post-strip
        structure ``run_simulations`` produces for equation models)."""
        params = {var: {"Intercept": 0} for var in self.stocks_and_auxiliaries}
        # Equation variables: an (initially empty) eq-params dict plus zeroed incoming links.
        for var in self.stocks_and_auxiliaries:
            if self._equation_evaluator and self._equation_evaluator.has_custom_equation(var):
                params[var][f"__eq_params_{var}__"] = {}
                for var_2 in self.variables:
                    if self.df_adj.loc[var, var_2] != 0:
                        params[var][var_2] = 0.0

        def place(entry, value):
            target = entry["target"]
            if entry["kind"] == "eq":
                params[target][f"__eq_params_{target}__"][entry["key"]] = float(value)
            elif entry["kind"] == "named":
                for var in self._named_param_users[entry["key"][1:]]:
                    params[var][f"__eq_params_{var}__"][entry["key"]] = float(value)
            else:  # link or interaction
                params[target][entry["key"]] = float(value)

        for entry, value in zip(free, np.asarray(vec, dtype=float)):
            place(entry, value)
        for entry in fixed:
            place(entry, entry["low"])
        return params

    def _gsa_intensities(self, intervention):
        """Resolve the intervention scenario to an intensity per intervention variable.

        ``None`` applies every intervention at unit intensity (a reference "full package"
        scenario, guaranteeing a non-degenerate outcome if any lever affects the VOI); a
        string applies that single lever; a dict maps lever names to intensities; an
        array-like is used as-is (length must match ``intervention_variables``).
        """
        n = len(self.intervention_variables)
        if intervention is None:
            return np.ones(n)
        if isinstance(intervention, str):
            if intervention not in self.intervention_variables:
                raise ValueError(f"'{intervention}' is not an intervention variable.")
            out = np.zeros(n)
            out[self.intervention_variables.index(intervention)] = 1.0
            return out
        if isinstance(intervention, dict):
            out = np.zeros(n)
            for name, val in intervention.items():
                if name not in self.intervention_variables:
                    raise ValueError(f"'{name}' is not an intervention variable.")
                out[self.intervention_variables.index(name)] = float(val)
            return out
        out = np.asarray(intervention, dtype=float)
        if out.shape != (n,):
            raise ValueError(f"intervention array must have length {n}, got {out.shape}.")
        return out

    def _gsa_evaluate(self, design, free, fixed, intensities, reducers, show_progress, relative_to=None):
        """Run the model once per design row and reduce to one scalar per reducer.

        ``reducers`` is a dict ``{label: callable(df_solution) -> float}``; returns a dict
        ``{label: np.ndarray}`` of outputs aligned with ``design``. With
        ``relative_to="baseline"`` each output is the reducer of the intervention run minus the
        reducer of a no-intervention run with the same parameters.
        """
        if relative_to not in (None, "baseline"):
            raise ValueError(f"relative_to must be None or 'baseline', got {relative_to!r}.")
        outputs = {label: np.empty(design.shape[0], dtype=float) for label in reducers}
        zero = np.zeros_like(np.asarray(intensities, dtype=float))
        iterator = range(design.shape[0])
        if show_progress:
            iterator = tqdm(iterator, desc="GSA: evaluating design")
        # Keep stdout quiet over thousands of runs (e.g. the analytical-solver notice).
        import io
        with self._magnitude_watch(), contextlib.redirect_stdout(io.StringIO()):
            for row_idx in iterator:
                params = self._gsa_params_from_vector(design[row_idx], free, fixed)
                df_sol = self.run_SDM_with_intervention_intensities(intensities, params)
                df_base = (self.run_SDM_with_intervention_intensities(zero, params)
                           if relative_to == "baseline" else None)
                for label, reduce_fn in reducers.items():
                    value = float(reduce_fn(df_sol))
                    if df_base is not None:
                        value -= float(reduce_fn(df_base))
                    outputs[label][row_idx] = value
        return outputs

    def run_GSA(self, variable_of_interest=None, *, method="sobol", bounds=None, n=1024,
                second_order=False, outcome="final", reducer=None, intervention=None,
                n_bootstrap=200, conf_level=0.95, clip=True, ci="bca", seed=None,
                show_progress=True, relative_to=None):
        """Global sensitivity analysis of the model's uncertain parameters.

        Three methods are available via ``method``:

        - ``"sobol"`` (default) — variance-based first-order ``S1`` and total-order ``ST`` Sobol
          indices from a Saltelli design (total runs ``n*(d+2)``), with BCa bootstrap confidence
          intervals projected onto ``[0, 1]``. Decomposes the *variance* of the outcome.
        - ``"delta"`` — Borgonovo's moment-independent ``delta`` measure (plus given-data ``S1``)
          from a Monte-Carlo ensemble of ``n`` runs. Captures the shift in the *whole output
          distribution*; non-negative by construction.
        - ``"pawn"`` — the CDF-based PAWN measure, also from an ``n``-run ensemble.

        The moment-independent methods (``delta``/``pawn``) do not suffer the near-zero
        negativity of variance decomposition and are cheaper (``n`` runs, not ``n*(d+2)``).

        Ranks the model's uncertain parameters by how much each drives the variance of an
        outcome (a variable of interest reduced over the simulated horizon), under a chosen
        intervention scenario. Returns first-order ``S1`` and total-order ``ST`` Sobol indices
        with bootstrap confidence intervals.

        Parameters
        ----------
        variable_of_interest : str or list[str], optional
            Outcome variable(s); defaults to ``self.variable_of_interest``. Ignored when a
            custom ``reducer`` is supplied.
        bounds : dict, optional
            ``{parameter_name: (low, high)}`` overrides of the default uniform priors. Names
            are ``"Target <- Source"`` (link coefficients), ``"Var | #1"`` (local equation
            parameters) and ``"#name"`` (named parameters). Use ``low == high`` to fix a
            parameter (excluded from the indices and reported).
        n : int
            Base Sobol sample size; total model runs = ``n*(d+2)`` (``n*(2d+2)`` with
            ``second_order``), where ``d`` is the number of free parameters. Use a power of two.
        outcome : {"final", "mean", "min", "max"}
            How to reduce each run's VOI trajectory to a scalar (default the final value).
        reducer : callable, optional
            ``callable(df_solution) -> float`` overriding ``outcome`` (and ``variable_of_interest``).
        intervention : None | str | dict | array-like
            Scenario under which the outcome is measured (see ``_gsa_intensities``).
        relative_to : None or "baseline"
            ``None`` (default) analyses the outcome itself; ``"baseline"`` analyses the
            intervention's effect, i.e. the outcome minus the outcome of a no-intervention run
            with the same parameters (doubles the number of runs).
        n_bootstrap, conf_level, seed : estimator controls.
        clip : bool
            If True (default), project the indices and their bootstrap confidence intervals
            onto the valid ``[0, 1]`` range, so an unimportant parameter reports ``S1 = 0`` with
            an asymmetric interval like ``[0, 0.05]`` rather than a spurious negative value. Pass
            ``clip=False`` for the raw estimates as a convergence diagnostic. See ``gsa.sobol_analyze``.

        Returns
        -------
        pandas.DataFrame or dict[str, pandas.DataFrame]
            One tidy frame per VOI (``parameter, S1, S1_low, S1_high, S1_conf, ST, ST_low,
            ST_high, ST_conf``, sorted by ``ST``); a single frame for one VOI or a custom
            reducer, else a ``{voi: frame}`` dict with asymmetric percentile CIs. The
            raw design and outputs are cached on ``self._gsa_design`` / ``self._gsa_outputs``,
            and the fixed parameters on ``self._gsa_fixed``.
        """
        from . import gsa

        if method not in ("sobol", "delta", "pawn"):
            raise ValueError(f"method must be 'sobol', 'delta' or 'pawn', got {method!r}.")

        free, fixed = self._gsa_parameter_spec(bounds)
        if not free:
            raise ValueError("No free parameters to analyse (all are fixed or the model has none).")
        names = [e["name"] for e in free]
        free_bounds = [(e["low"], e["high"]) for e in free]
        intensities = self._gsa_intensities(intervention)

        if method == "sobol":
            problem = gsa.sobol_problem(names, free_bounds)
            design = gsa.sobol_sample(problem, n, second_order=second_order, seed=seed)
        else:  # delta / pawn run on a plain Monte-Carlo (given-data) ensemble
            rng = np.random.default_rng(seed)
            lows = np.array([b[0] for b in free_bounds])
            highs = np.array([b[1] for b in free_bounds])
            design = rng.uniform(lows, highs, size=(int(n), len(free)))

        if reducer is not None:
            reducers = {"__reducer__": gsa.make_reducer(reducer=reducer)}
            voi_labels = ["__reducer__"]
        else:
            voi_list = variable_of_interest if variable_of_interest is not None else self.variable_of_interest
            if isinstance(voi_list, str):
                voi_list = [voi_list]
            reducers = {voi: gsa.make_reducer(outcome=outcome, voi=voi) for voi in voi_list}
            voi_labels = list(voi_list)

        outputs = self._gsa_evaluate(design, free, fixed, intensities, reducers, show_progress,
                                     relative_to=relative_to)

        self._gsa_design = pd.DataFrame(design, columns=names)
        self._gsa_outputs = {k: v for k, v in outputs.items()}
        self._gsa_fixed = {e["name"]: e["low"] for e in fixed}

        results = {}
        for label in voi_labels:
            if method == "sobol":
                df = gsa.sobol_analyze(problem, outputs[label], second_order=second_order,
                                       n_bootstrap=n_bootstrap, conf_level=conf_level, seed=seed,
                                       clip=clip, ci=ci)
            else:
                df = gsa.moment_independent(design, outputs[label], names, method=method,
                                            bounds=free_bounds, n_bootstrap=n_bootstrap,
                                            conf_level=conf_level, seed=seed)
            df.attrs["fixed_parameters"] = self._gsa_fixed
            results[label] = df

        if reducer is not None or len(voi_labels) == 1:
            return results[voi_labels[0]]
        return results

    def sample_outcomes(self, n=1000, *, variable_of_interest=None, bounds=None,
                        outcome="final", reducer=None, intervention=None, seed=None,
                        show_progress=True, relative_to=None):
        """Draw an ensemble of free-parameter samples and their scalar outcomes.

        Returns ``(X, y)`` where ``X`` is an ``n x d`` DataFrame of sampled free parameters
        (columns named as in :meth:`run_GSA`) and ``y`` is the length-``n`` outcome array,
        reduced exactly as in ``run_GSA`` (``relative_to="baseline"`` gives the effect of the
        intervention instead of the outcome itself). Convenience producer of the ensemble that
        :func:`discover_scenarios` consumes. Uses plain Monte-Carlo (uniform) draws, not a
        Sobol design, so the rows are independent.
        """
        from . import gsa

        free, fixed = self._gsa_parameter_spec(bounds)
        if not free:
            raise ValueError("No free parameters to sample (all are fixed or the model has none).")
        names = [e["name"] for e in free]
        rng = np.random.default_rng(seed)
        lows = np.array([e["low"] for e in free])
        highs = np.array([e["high"] for e in free])
        design = rng.uniform(lows, highs, size=(int(n), len(free)))
        intensities = self._gsa_intensities(intervention)

        if reducer is not None:
            reduce_fn = gsa.make_reducer(reducer=reducer)
        else:
            voi = variable_of_interest if variable_of_interest is not None else self.variable_of_interest
            if isinstance(voi, (list, tuple)):
                voi = voi[0]
            reduce_fn = gsa.make_reducer(outcome=outcome, voi=voi)

        outputs = self._gsa_evaluate(design, free, fixed, intensities,
                                     {"y": reduce_fn}, show_progress, relative_to=relative_to)
        return pd.DataFrame(design, columns=names), outputs["y"]

    def analytical_solution(self, t, x0, A, b):
        """ Analytical solution for a linear system of ODEs.
            We use the Pseudo-inverse because the regular inverse only works for non-singular matrices.
        """
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            print("Warning: Matrix A is singular. Falling back to numerical solver.")
            self.solve_analytically = 0  # Switch to numerical solver for future iterations

            # Use numerical solver instead for the current iteration
            return self._integrate(self.solve_sdm_linear, x0, (A, b), jac=self.jac_linear)

        I = np.identity(A.shape[0])
        A_inv_b = np.matmul(A_inv, b)

        sol = np.zeros((self.t_eval.shape[0], x0.shape[0]))
        sol[0, :] = x0  # Initial condition

        for i, t_i in enumerate(self.t_eval[1:]):  # Skip the first time point, which is x0
            exp_At = scipy.linalg.expm(A * t_i)
            sol[i + 1, :] = np.matmul(exp_At, x0) + np.matmul((exp_At - I), A_inv_b)
        return sol

    def solve_sdm_linear(self, t, x, A, b):
        """ Solve the linear system of differential equations representing the SDM.
        x: vector containing the stock and constant variables
        A: matrix of coefficients for the linear terms of len(x) in both dimensions.
        Outputs the derivative of x.
        """
        dx_dt = np.matmul(A, x) + b
        return dx_dt

    def jac_linear(self, t, x, A, b):
        """ Jacobian matrix is equal to A for linear systems
        """
        return A

    def compare_interventions_table(self, intervention_effects, n_bootstraps=200):
        """Compares interventions using the percentage of samples
        where one intervention is greater than the other, Cliff's Delta,
        and adds 95% bootstrapped confidence intervals for % Greater.

        ``intervention_effects`` is one VOI's entry of :meth:`get_intervention_effects`
        (by default baseline-subtracted effects); magnitudes are compared.
        """
        temp = []
        comparison_results = []

        for i in intervention_effects:
            for j in [i_e for i_e in intervention_effects if i_e not in temp]:
                if i != j:
                    samples_i = np.abs(intervention_effects[i])
                    samples_j = np.abs(intervention_effects[j])
                    differences = np.subtract(samples_i, samples_j)

                    greater_i = np.sum(differences > 0)
                    greater_j = np.sum(differences < 0)
                    cliff = (greater_i - greater_j) / len(differences)
                    percent_greater = round(greater_i * 100 / len(differences), 1)

                    # Bootstrap % Greater
                    bootstrapped_percents = []
                    for _ in range(n_bootstraps):
                        idx = np.random.choice(len(differences), size=len(differences), replace=True)
                        diff_sample = differences[idx]
                        greater_i_sample = np.sum(diff_sample > 0)
                        bootstrapped_percents.append(greater_i_sample * 100 / len(diff_sample))
                    lower = round(np.percentile(bootstrapped_percents, 2.5), 1)
                    upper = round(np.percentile(bootstrapped_percents, 97.5), 1)
                    ci_str = f"[{lower}, {upper}]"

                    # Store results
                    comparison_results.append([i, j, percent_greater, ci_str, round(cliff, 2)])

            temp.append(i)

        # Print combined table
        print("\nComparison Table (Percentage Greater, 95% CI, Cliff’s Delta):")
        print(tabulate(
            comparison_results,
            headers=["Intervention A", "Intervention B", "% Greater", "95% CI (% Greater)", "Cliff's Delta"],
            tablefmt="grid"
        ))

    def run_SA(self, outcome_var, int_var, cut_off_SA_importance=0.1, n_bootstraps=200, relative_to="baseline"):
        """ Run sensitivity analysis for the model parameters, either for a specific intervention (int_var) or over all interventions,
            and compute bootstrap confidence intervals for correlation coefficients.

        The effect is measured as in :meth:`get_intervention_effects`: by default the change at
        the final time relative to the no-intervention baseline run (``relative_to=None`` uses
        the raw final value). With ``outcome_var=None`` the effect is the mean absolute change
        over all model variables. Custom equation parameters are included (``p1->Var`` for local
        ``#1`` parameters, ``#name`` for named parameters).
        """
        if relative_to not in ("baseline", None):
            raise ValueError(f"relative_to must be 'baseline' or None, got {relative_to!r}.")
        baseline = getattr(self, 'df_sol_baseline_per_sample', None) if relative_to == "baseline" else None
        if relative_to == "baseline" and baseline is None:
            raise ValueError("No baseline runs found; call run_simulations() again, or pass relative_to=None.")

        if int_var is None:
            loop_over = self.intervention_variables
        else:
            loop_over = [int_var]

        # Build parameter names list, excluding special equation parameter keys
        param_names = []
        for target in self.param_samples[self.intervention_variables[0]]:
            for source in self.param_samples[self.intervention_variables[0]][target]:
                # Skip equation parameter arrays (they're handled separately)
                if source.startswith('__eq_params_'):
                    continue
                param_names.append(source + "->" + target)

        # Custom equation parameters: p<k>->Variable for local #k, #name for named ones
        eq_param_names = []
        if self._equation_evaluator:
            for var in self.stocks_and_auxiliaries:
                if self._equation_evaluator.has_custom_equation(var):
                    for idx in self.equations[var]['parameter_indices']:
                        eq_param_names.append(f"p{idx}->{var}")
            eq_param_names += [f"#{name}" for name in self.named_parameters]

        all_param_names = param_names + eq_param_names

        for i_v in loop_over:
            i = self.intervention_variables.index(i_v)
            for n in range(len(self.df_sol_per_sample)):
                # Collect regular parameters (scalars)
                params_curr = []
                for target in self.param_samples[i_v]:
                    for source in self.param_samples[i_v][target]:
                        if source.startswith('__eq_params_'):
                            continue
                        val = self.param_samples[i_v][target][source][n]
                        # Ensure it's a scalar
                        if hasattr(val, '__iter__') and not isinstance(val, str):
                            params_curr.extend(val)
                        else:
                            params_curr.append(float(val))

                # Collect equation parameters (dicts stored per sample)
                eq_params_curr = []
                named_values = {}
                if self._equation_evaluator:
                    for var in self.stocks_and_auxiliaries:
                        eq_key = f'__eq_params_{var}__'
                        if var in self.param_samples[i_v] and eq_key in self.param_samples[i_v][var]:
                            eq_item = self.param_samples[i_v][var][eq_key][n]
                            if isinstance(eq_item, dict):
                                for idx in self.equations[var]['parameter_indices']:
                                    eq_params_curr.append(float(eq_item[f'#{idx}']))
                                for name in self.equations[var].get('named_parameters', []):
                                    named_values.setdefault(name, float(eq_item[f'#{name}']))
                            # If array/list, flatten as before
                            elif hasattr(eq_item, '__iter__') and not isinstance(eq_item, str):
                                eq_params_curr.extend([float(x) for x in eq_item])
                            else:
                                eq_params_curr.append(float(eq_item))
                    eq_params_curr += [named_values.get(name, np.nan) for name in self.named_parameters]

                all_params = params_curr + eq_params_curr

                df_int = self.df_sol_per_sample[n][i]
                last = df_int.index[-1]
                if outcome_var is None:
                    cols = [c for c in df_int.columns if c != 'Time']
                    change = df_int.loc[last, cols].astype(float)
                    if baseline is not None:
                        change = change - baseline[n].loc[last, cols].astype(float)
                    eff_size = float(np.mean(np.abs(change.values)))
                else:
                    change = float(df_int.loc[last, outcome_var])
                    if baseline is not None:
                        change -= float(baseline[n].loc[last, outcome_var])
                    eff_size = abs(change)
                new_row = np.array(all_params + [float(eff_size)])

                df_SA_new = pd.DataFrame(new_row, index=all_param_names + ["Effect"]).T
                df_SA_new['intervention_variable'] = i_v

                if n == 0 and i_v == loop_over[0]:
                    df_SA = df_SA_new
                else:
                    df_SA = pd.concat([df_SA, df_SA_new], ignore_index=True)

        # Compute correlation, p-value, and bootstrapped confidence interval
        results = []
        for col in df_SA.columns:
            if col == "Effect" or col == "intervention_variable":
                continue
            # Skip Intercept parameters
            if "->" in col and col.split("->")[0] == "Intercept":
                continue

            # Spearman correlation and p-value
            rho, pval = scipy.stats.spearmanr(df_SA[col], df_SA["Effect"])

            # Bootstrap the Spearman correlation
            bootstrapped_corrs = []
            for _ in range(n_bootstraps):
                sample = df_SA.sample(n=len(df_SA), replace=True)
                r, _ = scipy.stats.spearmanr(sample[col], sample["Effect"])
                bootstrapped_corrs.append(r)

            lower = np.percentile(bootstrapped_corrs, 2.5)
            upper = np.percentile(bootstrapped_corrs, 97.5)

            mean_per_int = [scipy.stats.spearmanr(df_SA.loc[df_SA.intervention_variable==i_v, col],
                                                  df_SA.loc[df_SA.intervention_variable==i_v, "Effect"])[0]
                                                  for i_v in loop_over]

            if abs(rho) > cut_off_SA_importance:
                results.append([col, round(rho, 2),
                                #round(pval, 3),
                                f"[{round(lower, 2)}, {round(upper, 2)}]",
                                round(np.mean(mean_per_int), 2),
                                round(np.std(mean_per_int), 2)])

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x[1]), reverse=True)

        #headers = ["Variable", "Global rho", "p-value", "95% CI (bootstrap)"]
        headers = ["Link", "Rho", "95% CI (bootstrap)", "Mean Rho per Int", "SD Rho per Int"] #"p-value",

        print(tabulate(results, headers=headers, tablefmt="pretty"))

        # Also return the raw correlation dictionary and df for downstream use
        sorted_p_values = {row[0]: [row[1], row[2]] for row in results}
        return sorted_p_values, df_SA

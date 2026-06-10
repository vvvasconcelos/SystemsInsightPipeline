"""
Optimization module for System Dynamics Models.

This module provides intervention optimization functionality for SDM models,
including local and global optimization methods.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import qmc
import scipy
from copy import deepcopy
from tqdm import tqdm


class SDMOptimizer:
    """
    Optimizer for System Dynamics Models.
    
    This class provides methods to optimize intervention intensities for an SDM model
    subject to budget constraints.
    
    Args:
        sdm: An SDM instance to optimize
    
    Example:
        >>> from systemdynamics.sdm import SDM
        >>> from systemdynamics.optimizer import SDMOptimizer
        >>> sdm = SDM(settings)
        >>> optimizer = SDMOptimizer(sdm)
        >>> result = optimizer.optimize_intervention_intensities(params, costs)
    """
    
    def __init__(self, sdm):
        self.sdm = sdm
    
    def run_SDM_with_intervention_intensities(self, intervention_intensities, params):
        """ 
        Run the SDM with intervention intensities for each intervention variable for a given parameter set.
        
        Args:
            intervention_intensities (array-like): Vector of intervention intensities (magnitudes), one for each 
                                                   intervention variable in self.sdm.intervention_variables. 
                                                   Length must equal len(self.sdm.intervention_variables).
                                                   Intensities are always positive values representing the magnitude.
                                                   The direction (increase/decrease) is determined by the sign in 
                                                   self.sdm.intervention_strengths from the Excel file.
                                                   Example: [1, 0, 0, 0] applies only the first intervention at intensity 1.
                                                           [0.5, 0.5, 0, 0] applies first two interventions each at intensity 0.5.
                                                   If intervention_strengths[var] is negative, the variable will be decreased.
            params (dict): Parameter dictionary specifying the model structure 
        
        Returns:
            pd.DataFrame: Solution dataframe with all variables (stocks, constants, and auxiliaries) at each time step
        """
        # Validate input
        if len(intervention_intensities) != len(self.sdm.intervention_variables):
            raise ValueError(f"Length of intervention_intensities ({len(intervention_intensities)}) must match "
                           f"number of intervention variables ({len(self.sdm.intervention_variables)})")
        
        # Make a copy of params to avoid modifying the original
        params = deepcopy(params)
        
        # Initialize intervention arrays
        x0 = np.zeros(len(self.sdm.stocks), dtype=np.float64)
        constants_values = np.zeros(len(self.sdm.constants), dtype=np.float64)
        
        # Reset current intervention intensities for custom equations
        self.sdm._current_intervention_intensities = {}
        
        # Apply interventions based on intensities for each intervention variable
        # The intensity is always positive, but the direction is determined by the sign in intervention_strengths
        for var, intensity in zip(self.sdm.intervention_variables, intervention_intensities):
            if intensity == 0:  # Skip if no intervention
                continue
            
            # Get the sign from intervention_strengths (positive or negative intervention)
            intervention_sign = np.sign(self.sdm.intervention_strengths[var]) if self.sdm.intervention_strengths[var] != 0 else 1.0
            signed_intensity = intensity * intervention_sign
            
            if '+' not in var:  # Single factor intervention
                if var in self.sdm.stocks:
                    x0[self.sdm.stocks.index(var)] += signed_intensity
                elif var in self.sdm.constants:
                    constants_values[self.sdm.constants.index(var)] += signed_intensity
                else:
                    # Track intervention intensity for custom equations
                    self.sdm._current_intervention_intensities[var] = signed_intensity
                    # Only set intercept if no custom equation
                    if not (self.sdm._equation_evaluator and self.sdm._equation_evaluator.has_custom_equation(var)):
                        params[var]["Intercept"] = signed_intensity
            
            else:  # Double factor intervention
                var_1, var_2 = var.split('+')
                sign_1 = np.sign(self.sdm.intervention_strengths[var_1]) if self.sdm.intervention_strengths[var_1] != 0 else 1.0
                sign_2 = np.sign(self.sdm.intervention_strengths[var_2]) if self.sdm.intervention_strengths[var_2] != 0 else 1.0
                
                if var_1 in self.sdm.stocks:
                    x0[self.sdm.stocks.index(var_1)] += intensity * sign_1
                elif var_1 in self.sdm.constants:
                    constants_values[self.sdm.constants.index(var_1)] += intensity * sign_1
                else:
                    self.sdm._current_intervention_intensities[var_1] = intensity * sign_1
                    if not (self.sdm._equation_evaluator and self.sdm._equation_evaluator.has_custom_equation(var_1)):
                        params[var_1]["Intercept"] = intensity * sign_1
                
                if var_2 in self.sdm.stocks:
                    x0[self.sdm.stocks.index(var_2)] += intensity * sign_2
                elif var_2 in self.sdm.constants:
                    constants_values[self.sdm.constants.index(var_2)] += intensity * sign_2
                else:
                    self.sdm._current_intervention_intensities[var_2] = intensity * sign_2
                    if not (self.sdm._equation_evaluator and self.sdm._equation_evaluator.has_custom_equation(var_2)):
                        params[var_2]["Intercept"] = intensity * sign_2
        
        # Get system matrices A and b
        if self.sdm.interaction_terms:
            A = None
            b = None
        else:
            A, b = self.sdm.params_to_A_b(params, constants_values)
        
        # Run the SDM and return the solution
        df_sol = self.sdm.run_SDM(x0, constants_values, A, b, params)
        
        return df_sol

    def _compute_intervention_intensity_last(self, intensities_opt, costs):
        """
        Compute the last intervention intensity from the cost constraint.
        
        Args:
            intensities_opt (array-like): Intensities for the first n-1 interventions
            costs (array-like): Cost for each intervention variable
        
        Returns:
            float: Intensity for the last intervention
        """
        cost_burden = np.sum(np.array(intensities_opt) * np.array(costs[:-1]))
        intensity_last = (1.0 - cost_burden) / costs[-1]
        return intensity_last

    def _objective_function_intervention_opt(self, intensities_opt, params, costs, variable_of_interest, maximize=True):
        """
        Objective function for optimization.
        
        Args:
            intensities_opt (array-like): Intensities for the first n-1 interventions
            params (dict): Parameter dictionary
            costs (array-like): Cost for each intervention variable
            variable_of_interest (str): Variable to optimize for
            maximize (bool): If True, maximize the variable; if False, minimize it
        
        Returns:
            float: Objective value (negated if maximizing, for use with minimization algorithms)
        """
        # Compute the last intervention intensity
        intensity_last = self._compute_intervention_intensity_last(intensities_opt, costs)
        
        # Construct full intensity vector
        intervention_intensities = np.concatenate([intensities_opt, [intensity_last]])
        
        try:
            # Run the model
            df_sol = self.run_SDM_with_intervention_intensities(intervention_intensities, params)
            
            # Get the effect size at final time
            effect_size = df_sol.loc[self.sdm.t_eval[-1], variable_of_interest]
            
            # Return negative for maximization (since optimizer minimizes)
            # Return positive for minimization
            return -float(effect_size) if maximize else float(effect_size)
        except:
            # Return a large penalty if the model fails
            return 1e6


    #TODO: Consider whether local method is just an implementation of global with a specified sampling method and very low n_samples
    def optimize_intervention_intensities(self, params, costs, variable_of_interest=None, 
                                         bounds=None, initial_guess=None, method='global',
                                         threshold_effect=0.01, n_samples=None, maximize=True):
        """
        Optimize intervention intensities for a given parameter set subject to a cost constraint.
        
        Args:
            params (dict): Parameter dictionary for the model
            costs (array-like): Relative cost for each intervention variable
            variable_of_interest (str, optional): Variable to optimize for. If None, uses self.sdm.variable_of_interest[0]
            bounds (list, optional): Bounds for intensities. Default: [0, 5]
            initial_guess (array-like, optional): Initial guess for intensities (local method only)
            method (str): Optimization method - 'local' (L-BFGS-B, faster) or 'global' (SHGO, slower but finds multiple solutions). Default: 'global'
            threshold_effect (float): For global method - threshold to include near-optimal solutions. Default: 0.01
            n_samples (int, optional): For global method - target number of Sobol sampling points within the feasible region 
                                      (simplex constraint). Defaults to 30. The actual number of samples passed to SHGO will be 
                                      scaled up by n! (factorial of number of interventions) to account for the volume of the 
                                      feasible region relative to the unit hypercube. This assumes sampling methods generate 
                                      points uniformly in the full intervention space, with ~n_samples expected in the feasible subset.
                                      The final sample count is then snapped to the next power of 2 for Sobol balance properties.
                                      PERFORMANCE: Optimization pre-filters Sobol samples to only launch local optimizations from
                                      feasible points. For 4 interventions, n_samples=30 → ~40 local optimizations → ~1-3 minutes.
                                      Reduce n_samples to 10-15 for faster results (~30s), or use 'local' method for single solution (~1-5s).
            maximize (bool): If True, maximize the variable of interest; if False, minimize it. Default: True
        
        Returns:
            dict: For local method:
                - 'method': 'local'
                - 'optimized_intensities': Full intensity vector
                - 'effect_size': Optimal VOI value at final time
                - 'success': Whether optimization succeeded
                - 'n_evaluations': Number of function evaluations
                - 'optimization_result': Full scipy result object
            
            For global method:
                - 'method': 'global'
                - 'success': Whether optimization succeeded
                - 'n_equilibria': Number of near-optimal solutions found
                - 'equilibria': List of dicts with 'interventions', 'effect_size', 'total_cost'
                - 'best_effect_size': Best effect_size value found
                - 'optimization_result': Full scipy result object
        """
        if variable_of_interest is None:
            variable_of_interest = self.sdm.variable_of_interest[0]
        
        # Print a warning if the variable of interest is not in the model
        if variable_of_interest not in self.sdm.stocks_and_auxiliaries:
            warnings.warn(f"Variable of interest '{variable_of_interest}' not found in model variables.")

        
        costs = np.asarray(costs)
        n_interventions = len(self.sdm.intervention_variables)
        
        if len(costs) != n_interventions:
            raise ValueError(f"Length of costs ({len(costs)}) must match number of intervention variables ({n_interventions})")
        
        if method == 'local':
            return self._optimize_local(params, costs, variable_of_interest, bounds, initial_guess, maximize)
        elif method == 'global':
            # Calculate scaled and snapped sample count for global optimization
            import math
            if n_samples is None:
                n_samples = 30  # Default target number of feasible samples
            
            # Scale up to account for simplex volume: feasible region has volume 1/n! of unit hypercube
            volume_inverse_ratio = math.factorial(n_interventions)
            scaled_n_samples = int(n_samples * volume_inverse_ratio)
            
            # Snap to next power of 2 for Sobol balance properties
            def _next_power_of_two(k: int) -> int:
                return 1 << (max(1, k) - 1).bit_length()
            
            n_for_sobol = _next_power_of_two(scaled_n_samples)
            
            return self._optimize_global(params, costs, variable_of_interest, bounds, threshold_effect, n_for_sobol, maximize)
        else:
            raise ValueError(f"Method must be 'local' or 'global', got '{method}'")
    
    def _optimize_local(self, params, costs, variable_of_interest, bounds=None, initial_guess=None, maximize=True):
        """Local optimization using L-BFGS-B with equality constraint via elimination."""
        n_interventions = len(self.sdm.intervention_variables)
        
        # Set default bounds for n-1 variables
        if bounds is None:
            bounds = [(0, 5) for _ in range(n_interventions - 1)]
        
        # Set initial guess
        if initial_guess is None:
            initial_guess = np.ones(n_interventions - 1) / np.sum(costs[:-1])
        
        initial_guess = np.asarray(initial_guess[:n_interventions - 1])
        
        # Run optimization
        result = scipy.optimize.minimize(
            self._objective_function_intervention_opt,
            initial_guess,
            args=(params, costs, variable_of_interest, maximize),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'gtol': 1e-5}
        )
        
        # Extract optimized intensities
        optimized_intensities = np.concatenate([result.x, [self._compute_intervention_intensity_last(result.x, costs)]])
        
        # Evaluate objective at optimized point
        df_sol = self.run_SDM_with_intervention_intensities(optimized_intensities, params)
        optimal_effect_size = df_sol.loc[self.sdm.t_eval[-1], variable_of_interest]
        
        return {
            'method': 'local',
            'optimized_intensities': optimized_intensities,
            'effect_size': optimal_effect_size,
            'success': result.success,
            'n_evaluations': result.nfev,
            'optimization_result': result
        }
    
    def _optimize_global(
        self,
        params,
        costs,
        variable_of_interest,
        bounds=None,
        threshold_effect=0.01,
        n_for_sobol=1024,
        maximize=True,
        bounds_min=0,
        bounds_max=10,
    ):
        """Global optimization using multi-start approach with Sobol quasi-random sampling to find multiple near-optimal solutions.
        
        Implementation: Generates Sobol samples, pre-filters to feasible region (simplex constraint),
        then launches SLSQP local optimizations only from feasible starting points.
        This avoids wasting compute on ~96% infeasible starting points that SHGO would explore.
        
        PERFORMANCE: For 4 interventions with n_samples=30:
        - Generates 1024 Sobol samples → ~40 feasible → 40 local optimizations
        - Each local optimization: 10-50 function evaluations × ~0.1s = 1-5s
        - Total: 40-200 seconds expected (much faster than previous SHGO approach)
        
        Args:
            n_for_sobol (int): Pre-calculated, power-of-2 sample count for Sobol quasi-random exploration.
                              This value should be computed by optimize_intervention_intensities, which handles
                              scaling from user-specified n_samples and snapping to power of 2.
                              Sobol provides space-filling samples, more efficient than random sampling.
                              Only feasible samples are used as starting points for local optimization.
        """
        n_interventions = len(self.sdm.intervention_variables)
        costs_array = np.asarray(costs, dtype=float)

        # SHGO-only implementation in y-space: y_i = cost_i * intensity_i, sum(y) <= 1
        y_bounds = [(0.0, 1.0) for _ in range(n_interventions)]
        budget_constraint = {"type": "ineq", "fun": lambda y: 1.0 - np.sum(y)}

        def objective(y):
            try:
                y_arr = np.asarray(y, dtype=float)
                # Defensive penalty: constraint violation returns infinity
                if np.sum(y_arr) > 1.0 + 1e-10: 
                    return np.inf #TODO: Consider and test adding a dynamic value that brings the simulations closer to the line (e.g. proportional to sum(y_arr)). But we need a good limiting case where this is relevant
                # Get intensities from total investment, y
                intensities = np.divide(y_arr, costs_array, out=np.zeros_like(y_arr), where=costs_array > 0)
                df_sol = self.run_SDM_with_intervention_intensities(intensities, params)
                effect_size = float(df_sol[variable_of_interest].iloc[-1])
                # Non-finite result is penalized with infinity
                if not np.isfinite(effect_size):
                    return np.inf
                return -effect_size if maximize else effect_size
            except Exception:
                # Any evaluation failure returns infinity
                return np.inf

        # Generate Sobol samples and filter to FEASIBLE region before launching local optimizations
        # This avoids wasting compute on ~96% infeasible starting points
        sampler = qmc.Sobol(d=n_interventions, scramble=True)
        samples = sampler.random(n=n_for_sobol)
        
        # Filter to feasible points only (sum <= 1)
        feasible_mask = samples.sum(axis=1) <= 1.0
        feasible_samples = samples[feasible_mask]
        
        print(f"Sobol sampling: {n_for_sobol} total -> {len(feasible_samples)} feasible ({100*len(feasible_samples)/n_for_sobol:.1f}%)")
        
        if len(feasible_samples) == 0:
            return {
                "method": "global",
                "success": False,
                "message": "No feasible Sobol samples found",
                "optimization_result": None,
            }
        
        # Run local optimization from each feasible starting point
        local_results = []
        for i, y0 in enumerate(feasible_samples):
            try:
                local_res = scipy.optimize.minimize(
                    objective,
                    y0,
                    method='SLSQP',
                    bounds=y_bounds,
                    constraints=budget_constraint,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-3,
                    }
                )
                local_results.append(local_res)
            except Exception:
                continue
        
        if not local_results:
            return {
                "method": "global",
                "success": False,
                "message": "All local optimizations failed",
                "optimization_result": None,
            }
        
        # Find best result and collect near-optimal solutions
        valid_results = [r for r in local_results if r.success and np.isfinite(r.fun)]
        if not valid_results:
            valid_results = [r for r in local_results if np.isfinite(r.fun)]
        
        if not valid_results:
            return {
                "method": "global",
                "success": False,
                "message": "No valid solutions found",
                "optimization_result": local_results[0] if local_results else None,
            }
        
        # Best result
        best_result = min(valid_results, key=lambda r: r.fun)
        best_effect_size = (-best_result.fun) if maximize else (best_result.fun)
        
        # Mimic SHGO result structure for compatibility with downstream code
        result = type('obj', (object,), {
            'x': best_result.x,
            'fun': best_result.fun,
            'xl': [r.x for r in valid_results],
            'funl': [r.fun for r in valid_results],
            'success': True,
            'message': f'Multi-start optimization: {len(feasible_samples)} starts, {len(valid_results)} converged',
            'nfev': sum(r.nfev for r in local_results),
        })()

        # Collect all near-optimal solutions
        equilibria = []

        # result.xl: local minimizers (y-variables); result.funl: objective values at those points
        for y, val in zip(result.xl, result.funl):
            # Convert y directly to intensities: intensity_i = y_i / cost_i
            y_arr = np.asarray(y, dtype=float)
            intensities = np.divide(y_arr, costs_array, out=np.zeros_like(y_arr), where=costs_array > 0)

            current_effect_size = (-val) if maximize else (val)

            # Correct near-optimal test for both maximize and minimize
            if maximize:
                near_opt = (best_effect_size - current_effect_size) <= threshold_effect
            else:
                near_opt = (current_effect_size - best_effect_size) <= threshold_effect

            if near_opt:
                intervention_dict = {
                    name: round(float(intensity), 4)
                    for name, intensity in zip(self.sdm.intervention_variables, intensities)
                }
                equilibria.append(
                    {
                        "interventions": intervention_dict,
                        "intensities": intensities,
                        "effect_size": round(float(current_effect_size), 6),
                        "total_cost": round(float(np.dot(intensities, costs_array)), 4),
                    }
                )

        # Sort by effect_size (descending)
        equilibria.sort(key=lambda x: x["effect_size"], reverse=True)

        return {
            "method": "global",
            "success": True,
            "n_equilibria": len(equilibria),
            "equilibria": equilibria,
            "best_effect_size": float(best_effect_size),
            "optimization_result": result,
        }


    def optimize_across_parameter_samples(self, costs, variable_of_interest=None, 
                                         bounds=None, initial_guess=None, method='global',
                                         threshold_effect=0.01, n_samples=None, maximize=True):
        """
        Run optimization across all parameter samples to get optimized intervention intensities per sample.
        
        Args:
            costs (array-like): Relative cost for each intervention variable
            variable_of_interest (str, optional): Variable to optimize for. If None, uses self.sdm.variable_of_interest[0]
            bounds (list, optional): Bounds for intensities. Default: [0, 5]
            initial_guess (array-like, optional): Initial guess for intensities (local method only)
            method (str): Optimization method:
                - 'local': L-BFGS-B local optimization
                - 'global': SHGO with Sobol sampling
            threshold_effect (float): For global method - threshold for near-optimal solutions. Default: 0.01
            n_samples (int, optional): For global methods - controls population/sample size.
                                      If None, automatically calculated based on dimensionality.
            maximize (bool): If True, maximize the variable of interest; if False, minimize it. Default: True
        
        Returns:
            pd.DataFrame: For local method - dataframe with columns:
                - intensity_{intervention_var}: optimized intensity for each intervention
                - voi_effect_size: optimal VOI value at final time
                - optimization_success: whether optimization succeeded
                - n_evaluations: number of objective function evaluations
            
            For global method - dataframe with columns:
                - sample_idx: parameter sample index
                - equilibrium_idx: index of equilibrium within that sample
                - intensity_{intervention_var}: optimized intensity for each intervention
                - voi_effect_size: VOI value at final time
                - total_cost: total cost of intervention strategy
                - n_equilibria: total number of equilibria found for this sample
        """
        if variable_of_interest is None:
            variable_of_interest = self.sdm.variable_of_interest[0]
        
        results = []
        
        for sample_idx in tqdm(range(self.sdm.N), desc=f"Optimizing across parameter samples ({method})"):
            # Sample parameters for this sample
            params = self.sdm.sample_model_parameters()
            
            # Run optimization
            opt_result = self.optimize_intervention_intensities(
                params, costs, variable_of_interest, bounds, initial_guess, 
                method, threshold_effect, n_samples, maximize
            )
            
            if method == 'local':
                # Store single optimum per sample
                row = {}
                for i, int_var in enumerate(self.sdm.intervention_variables):
                    row[f'intensity_{int_var}'] = opt_result['optimized_intensities'][i]
                row['voi_effect_size'] = opt_result['effect_size']
                row['optimization_success'] = opt_result['success']
                row['n_evaluations'] = opt_result['n_evaluations']
                results.append(row)
            
            elif method == 'global':
                # Store all equilibria for this sample
                if opt_result['success']:
                    for eq_idx, equilibrium in enumerate(opt_result['equilibria']):
                        row = {'sample_idx': sample_idx, 'equilibrium_idx': eq_idx}
                        for int_var, intensity in equilibrium['interventions'].items():
                            row[f'intensity_{int_var}'] = intensity
                        row['voi_effect_size'] = equilibrium['effect_size']
                        row['total_cost'] = equilibrium['total_cost']
                        row['n_equilibria'] = opt_result['n_equilibria']
                        results.append(row)
                else:
                    # Record failure
                    row = {'sample_idx': sample_idx, 'equilibrium_idx': -1}
                    for int_var in self.sdm.intervention_variables:
                        row[f'intensity_{int_var}'] = np.nan
                    row['voi_effect_size'] = np.nan
                    row['total_cost'] = np.nan
                    row['n_equilibria'] = 0
                    results.append(row)
        
        # Convert to dataframe
        df_results = pd.DataFrame(results)
        
        return df_results

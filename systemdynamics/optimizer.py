"""
Optimization module for System Dynamics Models.

This module provides intervention optimization functionality for SDM models,
including local and global optimization methods.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo, differential_evolution
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
            n_samples (int, optional): For global method - number of SHGO samples using Sobol quasi-random sampling.
                                      If None (default), automatically set to 3 * (n_interventions - 1) to account for 
                                      effective dimensionality reduced by budget constraint. Use 'local' method for much faster optimization.
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
        
        # Auto-calculate n_samples based on dimensionality if not provided
        if n_samples is None:
            # With simplex parameterization, we have n_interventions-1 dimensions
            # Use 5x for reasonable Sobol coverage, minimum 10
            n_samples = max(10, 5 * (n_interventions-1))
        
        if method == 'local':
            return self._optimize_local(params, costs, variable_of_interest, bounds, initial_guess, maximize)
        elif method == 'global_de':
            return self._optimize_global_de(params, costs, variable_of_interest, bounds, threshold_effect, n_samples, maximize)
        elif method == 'global' or method == 'global_shgo':
            return self._optimize_global_shgo(params, costs, variable_of_interest, bounds, threshold_effect, n_samples, maximize)
        else:
            raise ValueError(f"Method must be 'local', 'global', 'global_de', or 'global_shgo', got '{method}'")
    
    
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
    
    def _optimize_global_de(self, params, costs, variable_of_interest, bounds=None, threshold_effect=0.01, n_samples=50, maximize=True, bounds_min=0, bounds_max=10):
        """Global optimization using differential evolution with simplex folding.
        
        The budget constraint is: sum(c_i * i_i) <= 1
        
        We reparametrize using y_i = c_i * i_i (cost-weighted intensities).
        The constraint becomes: sum(y_i) <= 1, which is the standard simplex.
        
        Simplex folding (sorting-based):
        - Sample u from [0,1]^(n-1)
        - Sort u to get order statistics: u_(1) <= u_(2) <= ... <= u_(n-1)
        - Compute spacings: y = [u_(1), u_(2)-u_(1), ..., 1-u_(n-1)]
        
        This is equivalent to "folding" the hypercube onto the simplex.
        In 2D (3 interventions): fold the unit square along the diagonal x=y.
        
        Mathematical fact: The spacings of uniform order statistics are 
        uniformly distributed on the simplex.
        
        Args:
            n_samples (int): Population size multiplier for differential evolution.
        """
        n_interventions = len(self.sdm.intervention_variables)
        costs_array = np.asarray(costs, dtype=np.float64)
        
        # Bounds for (n-1) values in [0, 1] - we need n-1 to generate n simplex coords
        simplex_bounds = [(0.0, 1.0) for _ in range(n_interventions - 1)]
        
        def fold_to_simplex(u):
            """Fold a point in [0,1]^(n-1) to the n-simplex using sorting.
            
            Given uniform u in [0,1]^(n-1), returns y on the n-simplex (sum(y) = 1).
            This is the standard transformation for uniform simplex sampling.
            """
            u_sorted = np.sort(u)
            # Compute differences: y_i = u_{(i)} - u_{(i-1)} where u_{(0)}=0, u_{(n-1)}=1
            # This produces n values from n-1 sorted values
            y = np.diff(np.concatenate([[0], u_sorted, [1]]))
            return y
        
        def y_to_intensities(y):
            """Convert cost-weighted values y to intensities.
            
            y_i = c_i * i_i, so i_i = y_i / c_i
            """
            # Avoid division by zero
            intensities = np.where(costs_array > 1e-10, y / costs_array, 0.0)
            return intensities
        
        def objective(u):
            """Objective function: fold u to simplex, convert to intensities, evaluate."""
            try:
                y = fold_to_simplex(u)
                intensities = y_to_intensities(y)
                df_sol = self.run_SDM_with_intervention_intensities(intensities, params)
                effect_size = df_sol.loc[self.sdm.t_eval[-1], variable_of_interest]
                return -float(effect_size) if maximize else float(effect_size)
            except:
                return 1e6
        
        # Run differential evolution
        # DE will explore [0,1]^(n-1) and we fold each point to the n-simplex
        popsize = max(5, n_samples // n_interventions) if n_samples else 5
        result = differential_evolution(objective, simplex_bounds, 
                                        popsize=popsize, maxiter=100, 
                                        tol=0.01, atol=0.01, seed=None,
                                        polish=True)
        
        if not result.success:
            return {
                'method': 'global',
                'success': False, 
                'message': 'Global optimization failed',
                'optimization_result': result
            }
        
        # Get the optimal solution
        best_effect_size = -result.fun if maximize else result.fun
        best_y = fold_to_simplex(result.x)
        best_intensities = y_to_intensities(best_y)
        
        intervention_dict = {name: round(float(intensity), 4) 
                           for name, intensity in zip(self.sdm.intervention_variables, best_intensities)}
        
        equilibria = [{
            'interventions': intervention_dict,
            'intensities': best_intensities,
            'effect_size': round(best_effect_size, 6),
            'total_cost': round(np.dot(best_intensities, costs_array), 4)
        }]
        
        return {
            'method': 'global_de',
            'success': True,
            'n_equilibria': 1,  # DE returns single optimum
            'equilibria': equilibria,
            'best_effect_size': best_effect_size,
            'optimization_result': result
        }

    def _optimize_global_shgo(self, params, costs, variable_of_interest, bounds=None, threshold_effect=0.01, n_samples=50, maximize=True, bounds_min=0, bounds_max=10):
        """Global optimization using SHGO with simplex folding.
        
        Same simplex folding as _optimize_global_de, but uses SHGO (Simplicial Homology 
        Global Optimization) instead of differential evolution.
        
        SHGO can find multiple local minima and provides more exhaustive search,
        but may be slower and can have numerical issues with Delaunay triangulation.
        
        Args:
            n_samples (int): Number of Sobol sampling points for SHGO.
        """
        n_interventions = len(self.sdm.intervention_variables)
        costs_array = np.asarray(costs, dtype=np.float64)
        
        # Bounds for (n-1) values in [0, 1]
        simplex_bounds = [(0.0, 1.0) for _ in range(n_interventions - 1)]
        
        def fold_to_simplex(u):
            """Fold a point in [0,1]^(n-1) to the n-simplex using sorting."""
            u_sorted = np.sort(u)
            y = np.diff(np.concatenate([[0], u_sorted, [1]]))
            return y
        
        def y_to_intensities(y):
            """Convert cost-weighted values y to intensities."""
            intensities = np.where(costs_array > 1e-10, y / costs_array, 0.0)
            return intensities
        
        def objective(u):
            """Objective function: fold u to simplex, convert to intensities, evaluate."""
            try:
                y = fold_to_simplex(u)
                intensities = y_to_intensities(y)
                df_sol = self.run_SDM_with_intervention_intensities(intensities, params)
                effect_size = df_sol.loc[self.sdm.t_eval[-1], variable_of_interest]
                return -float(effect_size) if maximize else float(effect_size)
            except:
                return 1e6
        
        # Run SHGO with Sobol sampling
        try:
            result = shgo(objective, simplex_bounds, n=n_samples, 
                         sampling_method='sobol', options={'ftol': 1e-4})
        except Exception as e:
            # SHGO can fail with Qhull errors - fall back to result with no success
            return {
                'method': 'global_shgo',
                'success': False,
                'message': f'SHGO failed: {str(e)}',
                'optimization_result': None
            }
        
        if not result.success:
            return {
                'method': 'global_shgo',
                'success': False, 
                'message': 'SHGO optimization failed',
                'optimization_result': result
            }
        
        # Get the optimal solution
        best_effect_size = -result.fun if maximize else result.fun
        best_y = fold_to_simplex(result.x)
        best_intensities = y_to_intensities(best_y)
        
        intervention_dict = {name: round(float(intensity), 4) 
                           for name, intensity in zip(self.sdm.intervention_variables, best_intensities)}
        
        equilibria = [{
            'interventions': intervention_dict,
            'intensities': best_intensities,
            'effect_size': round(best_effect_size, 6),
            'total_cost': round(np.dot(best_intensities, costs_array), 4)
        }]
        
        return {
            'method': 'global_shgo',
            'success': True,
            'n_equilibria': 1,
            'equilibria': equilibria,
            'best_effect_size': best_effect_size,
            'optimization_result': result
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
                - 'global' or 'global_de': Differential evolution with simplex folding (default, recommended)
                - 'global_shgo': SHGO with Sobol sampling and simplex folding
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
                    row['voi_outcome'] = np.nan
                    row['total_cost'] = np.nan
                    row['n_equilibria'] = 0
                    results.append(row)
        
        # Convert to dataframe
        df_results = pd.DataFrame(results)
        
        return df_results

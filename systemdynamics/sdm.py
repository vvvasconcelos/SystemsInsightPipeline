import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy
from copy import deepcopy
from tqdm import tqdm 
from tabulate import tabulate
from sympy.parsing.sympy_parser import parse_expr
import sympy as sym

class SDM:
    def __init__(self, s):
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

        self.num_pars_stocks = int((self.df_adj.loc[self.stocks, :] != 0).sum().sum())
        self.num_pars_auxiliaries = int((self.df_adj.loc[self.auxiliaries, :] != 0).sum().sum())

        stock_indices = [s.variables.index(stock) for stock in self.stocks]
        aux_indices = [s.variables.index(aux) for aux in self.auxiliaries]
        # self.num_pars_int_stocks = int(self.interactions_matrix[np.ix_(stock_indices,
        #                                                            np.arange(self.interactions_matrix.shape[1]),
        #                                                            np.arange(self.interactions_matrix.shape[2]))].sum().sum())
        # self.num_pars_int_auxiliaries = int(self.interactions_matrix[np.ix_(aux_indices,
        #                                                                 np.arange(self.interactions_matrix.shape[1]),
        #                                                                 np.arange(self.interactions_matrix.shape[2]))].sum().sum())

        self.num_pars_int_stocks = int(np.count_nonzero(self.interactions_matrix[np.ix_(stock_indices,
                                            np.arange(self.interactions_matrix.shape[1]),
                                            np.arange(self.interactions_matrix.shape[2]))]))
        
        self.num_pars_int_auxiliaries = int(np.count_nonzero(self.interactions_matrix[np.ix_(aux_indices,
                                            np.arange(self.interactions_matrix.shape[1]),
                                            np.arange(self.interactions_matrix.shape[2]))]))

        # Set the SDM simulation timesteps to store 
        self.solver = 'LSODA'  # 'LSODA' automatically switches between stiff and non-stiff methods since stiffness is not always known.
        self.t_span = [0.0, s.t_end] 
        self.t_eval = np.array(self.t_span)  # Only evaluate final time point for analytical solution

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def run_simulations(self, progress_callback=None):
        """ Run the simulations for N iterations for all the specified interventions
        """
        df_sol_per_sample = []  # List for storing the solution dataframes
        param_samples = {var : {} for var in self.intervention_variables}  # Dictionary for storing the parameters across samples

        for num in tqdm(range(self.N), desc="Running Simulations"):  # Iterate over the number of samples
            df_sol = []

            params_i = self.sample_model_parameters()  # Sample model parameters

            for i, var in enumerate(self.intervention_variables):
                # Set the initial condition for the stocks to zero
                x0 = np.zeros(len(self.stocks), dtype=np.float64)  # By default no intervention on a stock or constant
                constants_values = np.zeros(len(self.constants), dtype=np.float64)  # By default no intervention on a constant 

                params = deepcopy(params_i)  # Copy the parameters to avoid overwriting the original parameters

                if '+' not in var:  # Single factor intervention
                    if var in self.stocks:
                        x0[self.stocks.index(var)] += self.intervention_strengths[var] 
                    elif var in self.constants:
                        constants_values[self.constants.index(var)] += self.intervention_strengths[var]
                    else:
                        params[var]["Intercept"] = self.intervention_strengths[var]

                else:  # Double factor intervention
                    var_1, var_2 = var.split('+')
                    if var_1 in self.stocks: 
                        x0[self.stocks.index(var_1)] += (1/2) * self.intervention_strengths[var_1]
                    elif var_1 in self.constants: 
                        constants_values[self.constants.index(var_1)] += (1/2) * self.intervention_strengths[var_1]
                    else: 
                        params[var_1]["Intercept"] = (1/2) * self.intervention_strengths[var_1]

                    if var_2 in self.stocks:
                        x0[self.stocks.index(var_2)] += (1/2) * self.intervention_strengths[var_2]
                    elif var_2 in self.constants: 
                        constants_values[self.constants.index(var_2)] += (1/2) * self.intervention_strengths[var_2]
                    else:
                        params[var_2]["Intercept"] = (1/2) * self.intervention_strengths[var_2]

                if self.interaction_terms:
                    #K = self.params_to_K(params)
                    A = None
                    b = None
                else: # Get the system matrices A and b
                    A, b = self.params_to_A_b(params, constants_values)

                df_sol_per_intervention = self.run_SDM(x0, constants_values, A, b, params)
                df_sol += [df_sol_per_intervention]

                # Store the model parameters
                if num == 0: 
                    param_samples[var] = {target : {source : [params[target][source]] for source in params[target]} for target in params}
                else:
                    for target in params:
                        for source in params[target]:
                            param_samples[var][target][source] += [params[target][source]]

            df_sol_per_sample += [df_sol]

            # If a progress callback is provided, update progress (for Streamlit app)
            # if progress_callback:
            #     progress_callback(num + 1, self.N)
    
        self.df_sol_per_sample = df_sol_per_sample
        self.param_samples = param_samples
        return df_sol_per_sample, param_samples

    def run_SDM(self, x0, constants_values, A, b, params):
        """ Run the SDM and return a dataframe with all the variables at every time step, including auxiliaries.
        """
        if self.interaction_terms:
            # solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
            #                     t_eval=self.t_eval, jac=self.jac,
            #                     method=self.solver, rtol=1e-6, atol=1e-6).y.T
            solution = solve_ivp(self.dz_dt_int, self.t_span, x0, args=(constants_values, params),
                                 t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6).y.T       
        else:  # Linear system
            if self.solve_analytically: 
                solution = self.analytical_solution(self.t_eval[:, None], x0, A, b)
            else:
                solution = solve_ivp(self.solve_sdm_linear, self.t_span, x0, args=(A, b),
                                   t_eval=self.t_eval, jac=self.jac_linear,
                                   method=self.solver, rtol=1e-6, atol=1e-6).y.T

        if np.sum(solution > 10):
            print("Warning: Solution has values larger than 10. The maximum parameter values may be too large.")

        df_sol = pd.DataFrame(solution, columns=self.stocks, index=self.t_eval)
        df_sol["Time"] = df_sol.index

        # Add constants to the dataframe
        for i, const in enumerate(self.constants):
            df_sol.loc[:, const] = constants_values[i]

        # Evaluate auxiliaries using the parameter dictionary and add to dataframe
        params_aux_only = {var : params[var] for var in self.auxiliaries}
        df_sol_with_aux = self.evaluate_auxiliaries(params_aux_only, df_sol)
        return df_sol_with_aux
    
    def dz_dt_int(self, t, z, constants_values, params):
        """Compute dz/dt for nonlinear SDM with interaction terms using only NumPy."""
        # Combine all variables into one vector
        vals = {var: 0.0 for var in self.variables}
        for s, v in zip(self.stocks, z):
            vals[s] = v
        for c, v in zip(self.constants, constants_values):
            vals[c] = v

        # --- Compute auxiliaries ---
        for aux in self.auxiliaries:
            total = 0.0
            for pred, val in params[aux].items():
                if pred == 'Intercept':
                    total += val
                elif '*' in pred:
                    v1, v2 = [p.strip() for p in pred.split('*')]
                    total += val * vals[v1] * vals[v2]
                else:
                    total += val * vals[pred]
            vals[aux] = total

        # --- Compute stock derivatives ---
        dz = np.zeros(len(self.stocks))
        for i, s in enumerate(self.stocks):
            total = 0.0
            for pred, val in params[s].items():
                if pred == 'Intercept':
                    total += val
                elif '*' in pred:
                    v1, v2 = [p.strip() for p in pred.split('*')]
                    total += val * vals[v1] * vals[v2]
                else:
                    total += val * vals[pred]
            dz[i] = total
        return dz

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
        """
        """
        # First sort the auxiliaries based on mutual dependencies
        auxiliaries_sorted = []
        deps = {a: [d for d in params.get(a, {}) if d in self.auxiliaries] for a in self.auxiliaries}
        warning_count = 0
        while deps:
            ready = [a for a, ds in deps.items() if not ds]
            auxiliaries_sorted.extend(ready)
            for r in ready: deps.pop(r)
            for ds in deps.values():
                ds[:] = [d for d in ds if d not in ready]
            warning_count += 1
            if warning_count > 100:  # Arbitrary large number to prevent infinite loops
                raise ValueError("Warning: Possible circular dependency among auxiliaries. Check model structure.")
        return auxiliaries_sorted

    def evaluate_auxiliaries(self, params, df_sol):
        """ Evaluate the auxiliary variables at each time step.
        Input: original parameter dictionary with auxiliary terms, and the solution dataframe
        Output: dataframe with added auxiliary values at each time step
        """
        if self.auxiliaries_sorted == []: # If not already sorted
            self.auxiliaries_sorted = self.sort_auxiliaries(params)
    
        df_sol_with_auxiliaries = df_sol.copy()
        for aux in self.auxiliaries_sorted:
            df_sol_with_auxiliaries[aux] = np.nan # Initialize column with NaNs
            for t in self.t_eval:
                aux_value = 0
                for origin in params[aux]:
                    if origin == "Intercept":
                        aux_value += params[aux][origin]
                    else:
                        if "*" in origin:  # Interaction term
                            origin_1 = origin.split(" * ")[0]
                            origin_2 = origin.split(" * ")[1]
                            aux_value += params[aux][origin] * df_sol_with_auxiliaries.loc[t, origin_1] * df_sol_with_auxiliaries.loc[t, origin_2]
                        else:  # Not an interaction term
                            aux_value += params[aux][origin] * df_sol_with_auxiliaries.loc[t, origin]
                df_sol_with_auxiliaries.loc[t, aux] = aux_value

        return df_sol_with_auxiliaries

    def get_intervention_effects(self):
        """ Obtain intervention effects from a dataframe with model simulation results.
        """
        intervention_effects_per_voi = {voi : {} for voi in self.variable_of_interest}

        for voi in self.variable_of_interest:
            # Get the intervention effects at the final time point (compared to the implicit counterfactual where everything stays zero)
            intervention_effects = {i_v : [self.df_sol_per_sample[n][i].loc[self.t_eval[-1], voi] 
                                    for n in range(self.N)] for i, i_v in enumerate(self.intervention_variables)}
            # Sort the dictionary by the median intervention effect
            intervention_effects = dict(sorted(intervention_effects.items(),
                                            key=lambda item: np.median(np.abs(item[1])), reverse=True))

            self.intervention_effects = intervention_effects
            intervention_effects_per_voi[voi] = intervention_effects
        
        return intervention_effects_per_voi

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

    def sample_model_parameters(self): #, intervention_auxiliaries=None):
        """ Sample from the model parameters using a bounded uniform distribution. 
            The possible parameters are given by the adjacency and interactions matrices.
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
    
        for i, var in enumerate(self.variables):
            # Intercept
            if var in self.stocks_and_auxiliaries:
                params[var]["Intercept"] = 0

            # Pairwise interactions
            for j, var_2 in enumerate(self.variables):
                if self.df_adj.loc[var, var_2] != 0:
                    if self.df_adj.loc[var, var_2] == -999:
                        if var in self.stocks:  # If the variable is a stock
                            params[var][var_2] = (sample_pars_stocks[par_count_stocks] * 2) - self.parameter_value_stocks  # Uniform[-self.max_parameter_value, self.max_parameter_value]
                            par_count_stocks += 1
                        else:  # If the variable is an auxiliary
                            params[var][var_2] = (sample_pars_auxiliaries[par_count_auxiliaries] * 2) - self.parameter_value_aux  # Uniform[-self.max_parameter_value_aux, self.max_parameter_value_aux]
                            par_count_auxiliaries += 1
                    else:   
                        if var in self.stocks:  # If the variable is a stock
                            params[var][var_2] = self.df_adj.loc[var, var_2] * sample_pars_stocks[par_count_stocks]
                            par_count_stocks += 1
                        else:  # If the variable is an auxiliary
                            params[var][var_2] = self.df_adj.loc[var, var_2] * sample_pars_auxiliaries[par_count_auxiliaries]
                            par_count_auxiliaries += 1

                # 2nd-order interaction terms
                if self.interaction_terms:
                    for k, var_3 in enumerate(self.variables):
                        if self.interactions_matrix[i, j, k] != 0:
                            if self.df_adj.loc[var, var_2] == -999:
                                if var in self.stocks:  # If the variable is a stock
                                    params[var][var_2 + " * " + var_3] = (sample_pars_int_stocks[par_int_count_stocks] * 2) - self.parameter_value_stocks  # Uniform[-self.max_parameter_value, self.max_parameter_value]
                                    par_int_count_stocks += 1
                                elif var in self.auxiliaries:  # If the variable is an auxiliary
                                    params[var][var_2 + " * " + var_3] = (sample_pars_int_auxiliaries[par_int_count_auxiliaries] * 2) - self.parameter_value_aux
                                    par_int_count_auxiliaries += 1
                            else:   
                                if var in self.stocks: # If the variable is a stock
                                    params[var][var_2 + " * " + var_3] = self.interactions_matrix[i, j, k] * sample_pars_int_stocks[par_int_count_stocks]
                                    par_int_count_stocks += 1
                                if var in self.auxiliaries: # If the variable is an auxiliary
                                    params[var][var_2 + " * " + var_3] = self.interactions_matrix[i, j, k] * sample_pars_int_auxiliaries[par_int_count_auxiliaries]
                                    par_int_count_auxiliaries += 1   
        self.params = params
        return params

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
            return solve_ivp(self.solve_sdm_linear, self.t_span, x0, args=(A, b),
                                   t_eval=self.t_eval, jac=self.jac_linear,
                                   method=self.solver, rtol=1e-6, atol=1e-6).y.T

            # A_inv = np.linalg.pinv(A)  # Pseudo-inverse for singular matrices

        I = np.identity(A.shape[0])
        A_inv_b = np.matmul(A_inv, b)

        sol = np.zeros((self.t_eval.shape[0], x0.shape[0]))
        sol[0, :] = x0  # Initial condition

        for i, t_i in enumerate(self.t_eval[1:]):  # Skip the first time point, which is x0
            exp_At = scipy.linalg.expm(A * t_i)
            sol[i + 1, :] = np.matmul(exp_At, x0) + np.matmul((exp_At - I), A_inv_b)
        return sol

    def solve_sdm(self, t, x, A, K, b):
        """ Solve the system of differential equations representing the SDM.
        x: vector containing the stock and constant variables
        A: matrix of coefficients for the linear terms of len(x) in both dimensions.
        K: 3rd order tensor of coefficients for the interaction terms of len(x) in all three dimensions.
        Outputs the derivative of x.
        """
        Kx = np.matmul(K, x) 
        dx_dt = np.matmul(A, x) + np.matmul(Kx, x) + b
        return dx_dt

    def solve_sdm_linear(self, t, x, A, b):
        """ Solve the linear system of differential equations representing the SDM.
        x: vector containing the stock and constant variables
        A: matrix of coefficients for the linear terms of len(x) in both dimensions.
        Outputs the derivative of x.
        """
        dx_dt = np.matmul(A, x) + b
        return dx_dt

    def jac(self, t, x, A, K, b):
        """ Jacobian matrix, depends on A and K, not b (which is a constant vector)
        """
        return A + 2 * np.matmul(K, x)

    def jac_linear(self, t, x, A, b):
        """ Jacobian matrix is equal to A for linear systems
        """
        return A

    def compare_interventions_table(self, intervention_effects, n_bootstraps=200):
        """Compares interventions using the percentage of samples 
        where one intervention is greater than the other, Cliff's Delta,
        and adds 95% bootstrapped confidence intervals for % Greater.
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
        
    def run_SA(self, outcome_var, int_var, cut_off_SA_importance=0.1, n_bootstraps=200):
        """ Run sensitivity analysis for the model parameters, either for a specific intervention (int_var) or over all interventions,
            and compute bootstrap confidence intervals for correlation coefficients.
        """  
        if int_var is None:
            loop_over = self.intervention_variables
        else:
            loop_over = [int_var]

        param_names = self.flatten([[source + "->" + target for source in self.param_samples[self.intervention_variables[0]][target]]
                                    for target in self.param_samples[self.intervention_variables[0]]])

        for i_v in loop_over:
            i = self.intervention_variables.index(i_v)
            for n in range(self.N):
                params_curr = self.flatten([[self.param_samples[i_v][target][source][n]
                                            for source in self.param_samples[i_v][target]]
                                            for target in self.param_samples[i_v]])

                if outcome_var is None:
                    eff_size = self.df_sol_per_sample[n][i].loc[self.df_sol_per_sample[n][i].Time == self.t_eval[-1], :].abs().mean().mean()
                    new_row = np.array(params_curr + [float(eff_size)])
                else:
                    eff_size = abs(self.df_sol_per_sample[n][i].loc[self.df_sol_per_sample[n][i].Time == self.t_eval[-1], outcome_var])
                    new_row = np.array(params_curr + [float(eff_size.iloc[0])])

                df_SA_new = pd.DataFrame(new_row, index=param_names + ["Effect"]).T
                df_SA_new['intervention_variable'] = i_v

                if n == 0 and i_v == loop_over[0]:
                   df_SA = df_SA_new
                else:
                    df_SA = pd.concat([df_SA, df_SA_new], ignore_index=True)

        # Compute correlation, p-value, and bootstrapped confidence interval
        results = []
        for col in df_SA.columns:
            if col == "Effect" or col.split("->")[0] == "Intercept" or col == "intervention_variable":
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
                                                  df_SA.loc[df_SA.intervention_variable==i_v, "Effect"]).statistic 
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

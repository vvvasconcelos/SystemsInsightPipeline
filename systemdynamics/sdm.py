import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy
import re
from copy import deepcopy
from sympy.parsing.sympy_parser import parse_expr
from scipy.stats import halfnorm
import sympy as sym
import networkx as nx
from tqdm import tqdm 
from tabulate import tabulate

class SDM:
    def __init__(self, s):
        self.df_adj = s.df_adj
        self.df_adj_incl_interactions = s.df_adj_incl_interactions
        self.N = s.N
        self.interactions_matrix = s.interactions_matrix
        self.interaction_terms = s.interaction_terms
        self.solve_analytically = s.solve_analytically
        self.stability_analysis = True
        
        self.stocks_and_auxiliaries = s.stocks_and_auxiliaries
        self.stocks_and_constants = s.stocks_and_constants
        self.constants = s.constants
        self.auxiliaries = s.auxiliaries
        self.simulate_interventions = s.simulate_interventions
        self.variables = s.variables
        self.stocks = s.stocks
        self.parameter_value_stocks = s.parameter_value_stocks
        self.parameter_value_aux = s.parameter_value_aux
        self.variable_of_interest = s.variable_of_interest
        self.intervention_variables = s.intervention_variables
        self.prior = "uniform"  # s.prior
        self.intervention_effects = s.intervention_effects
        np.random.seed(s.seed)  # Set seed for reproducibility

        self.num_pars_stocks = int((self.df_adj.loc[self.stocks, :] != 0).sum().sum())
        self.num_pars_auxiliaries = int((self.df_adj.loc[self.auxiliaries, :] != 0).sum().sum())

        stock_indices = [s.variables.index(stock) for stock in self.stocks]
        aux_indices = [s.variables.index(aux) for aux in self.auxiliaries]
        self.num_pars_int_stocks = int(self.interactions_matrix[np.ix_(stock_indices,
                                                                   np.arange(self.interactions_matrix.shape[1]),
                                                                   np.arange(self.interactions_matrix.shape[2]))].sum().sum())
        self.num_pars_int_auxiliaries = int(self.interactions_matrix[np.ix_(aux_indices,
                                                                        np.arange(self.interactions_matrix.shape[1]),
                                                                        np.arange(self.interactions_matrix.shape[2]))].sum().sum())

        # self.num_pars_int_stocks = int(np.abs(self.interactions_matrix[self.stocks, :, :]).sum().sum())
        # self.num_pars_int_auxiliaries = int(np.abs(self.interactions_matrix[self.auxiliaries, :, :]).sum().sum())

        # Set the SDM simulation timesteps to store 
        s.dt = 1  # Time step for the simulation, only for non-analytical solutions #TO DO ADJUST
        s.t_eval = np.array(np.array([0.0] + list(np.linspace(0, s.t_end,
                                                            int(s.t_end/s.dt) + 1)[1:])))
        
        # If solving the system numerically, set the solver
        #s.solver = 'LSODA'  # 'LSODA' automatically switches between stiff and non-stiff methods since stiffness is not always known.
        self.solver = 'LSODA'
        self.t_eval = s.t_eval
        self.t_span = [s.t_eval[0], s.t_eval[-1]]
        if self.solve_analytically:
            self.t_eval = np.array(self.t_span)  # Only evaluate final time point for analytical solution

        # Run tests
        # self.test_vectorized_eqs()  # Call the test_vectorized_eqs function when the class is loaded
        # self.test_get_link_scores()  # Call the test_get_link_scores function when the class is loaded

        #if s.interaction_terms == 0:
        #    self.test_with_linear_model()  # Test whether analytical solution and numerical solution match

        #if s.setting_name == "Sleep" and s.variable_of_interest == "Depressive_symptoms":
        #    self.test_with_sleep_depression_model() # Call the test_with_sleep_depression_model function when the class is loaded

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

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
        df_SA = pd.DataFrame(columns=param_names + ["Effect"])

        for i_v in loop_over:
            i = self.intervention_variables.index(i_v)
            for n in range(self.N):
                params_curr = self.flatten([[self.param_samples[i_v][target][source][n]
                                            for source in self.param_samples[i_v][target]]
                                            for target in self.param_samples[i_v]])
                if outcome_var is not None:
                    eff_size = abs(self.df_sol_per_sample[n][i].loc[self.df_sol_per_sample[n][i].Time == self.t_eval[-1], outcome_var])
                    new_row = np.array(params_curr + [float(eff_size.iloc[0])])
                else:
                    eff_size = self.df_sol_per_sample[n][i].loc[self.df_sol_per_sample[n][i].Time == self.t_eval[-1], :].abs().mean().mean()
                    new_row = np.array(params_curr + [float(eff_size)])

                df_SA_new = pd.DataFrame(new_row, index=param_names + ["Effect"]).T
                df_SA = pd.concat([df_SA, df_SA_new], ignore_index=True)

        # Compute correlation, p-value, and bootstrapped confidence interval
        results = []
        for col in df_SA.columns:
            if col == "Effect" or col.split("->")[0] == "Intercept":
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

            if abs(rho) > cut_off_SA_importance:
                results.append([col, round(rho, 2), round(pval, 3), f"[{round(lower, 2)}, {round(upper, 2)}]"])

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x[1]), reverse=True)

        headers = ["Variable", "Spearman correlation", "p-value", "95% CI (bootstrap)"]
        print(tabulate(results, headers=headers, tablefmt="pretty"))

        # Also return the raw correlation dictionary and df for downstream use
        sorted_p_values = {row[0]: [row[1], row[2]] for row in results}
        return sorted_p_values, df_SA

    def run_simulations(self, progress_callback=None):
        """ Run the simulations for N iterations for all the specified interventions
        """
        df_sol_per_sample = []  # List for storing the solution dataframes
        #df_sol_per_sample_no_int = []   # List for storuing the solution dataframes without interventions
        param_samples = {var : {} for var in self.intervention_variables}  # Dictionary for storing the parameters across samples

        if self.stability_analysis:
            eigenvalues_real = np.zeros((self.N, len(self.stocks_and_constants)))
            eigenvalues_imag = np.zeros((self.N, len(self.stocks_and_constants)))
            eig_val_vec = {"Eigenvalues": [], "Eigenvectors" : []}

        for num in tqdm(range(self.N), desc="Running Simulations"):  # Iterate over the number of samples
            df_sol = []

            params_i = self.sample_model_parameters() #s.intervention_auxiliaries)  # Sample model parameters

            for i, var in enumerate(self.intervention_variables):
                # Set the initial condition for the stocks to zero
                x0 = np.zeros(len(self.stocks_and_constants), order='F')  # By default no intervention on a stock or constant (initialized in equilibrium)
                #intervention_auxiliaries = {}  # By default no intervention on an auxiliary
                params = deepcopy(params_i)  # Copy the parameters to avoid overwriting the original parameters

                if '+' in var:  # Double factor intervention
                    var_1, var_2 = var.split('+')
                    if var_1 in self.stocks_and_constants:  # Intervention on a stock or constant (first intervention variable)
                        x0[self.stocks_and_constants.index(var_1)] += (1/2)*self.intervention_effects[var_1]  # Increase the (baseline) value of the stock/constant by 1/2
                    else:  # Intervention on an auxiliary
                        params[var_1]["Intercept"] = (1/2)*self.intervention_effects[var_1]
                    if var_2 in self.stocks_and_constants:
                        x0[self.stocks_and_constants.index(var_2)] += (1/2)*self.intervention_effects[var_2] 
                    else:
                        params[var_2]["Intercept"] = (1/2)*self.intervention_effects[var_2]
                else:  # Single factor intervention
                    if var in self.stocks_and_constants:  # Intervention on a stock or constant (only variable)
                        x0[self.stocks_and_constants.index(var)] += self.intervention_effects[var]  # Increase the (baseline) value of the stock/constant by 1
                    else:  # Intervention on an auxiliary (only variable)
                        params[var]["Intercept"] = self.intervention_effects[var]

                new_params = self.make_equations_auxiliary_independent(params)  # Remove auxiliaries from the equations
                if np.sum([[1 for par in new_params[st] if par in self.auxiliaries] for st in new_params]) > 0:
                    raise(Exception('Some parameters are defined for auxiliaries. This means the process of making equations auxiliary independent failed.',
                                    'Likely because of a feedback loop with only auxiliaries. Please ensure that all feedback loops contain at least one stock.'))
                A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries
                
                if self.stability_analysis and i == 0:  # The initial conditions do not matter for the linear stability analysis
                    eigenvalues = np.linalg.eigvals(self.get_A_and_K_matrices()[0])
                    eigenvalues = np.linalg.eigvals(self.get_A_and_K_matrices()[0])
                    eigenvectors = np.linalg.eig(self.get_A_and_K_matrices()[0])[1]
                    eigenvalues_real[num, :] = eigenvalues.real
                    eigenvalues_imag[num, :] = eigenvalues.imag
                    eig_val_vec["Eigenvalues"] += [eigenvalues]
                    eig_val_vec["Eigenvectors"] += [eigenvectors]

                df_sol_per_intervention = self.run_SDM(x0, A, K, b)
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
            if progress_callback:
                progress_callback(i + 1, self.N)
    
        self.df_sol_per_sample = df_sol_per_sample
        self.param_samples = param_samples
        return df_sol_per_sample, param_samples, eig_val_vec #df_stability

    def get_intervention_effects(self):
        """ Obtain intervention effects from a dataframe with model simulation results.
        """

        intervention_effects_per_voi = {voi : {} for voi in self.variable_of_interest}

        for voi in self.variable_of_interest:
            intervention_effects = {i_v : [(self.df_sol_per_sample[n][i].loc[self.t_eval[-1], voi] -
                                    self.df_sol_per_sample[n][i].loc[0, voi]) 
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
        
        # Return the original column names exactly as they are
        return list(sorted_columns)

    def run_loops_that_matter(self, int_var=None): 
        """ Calculate link and loop scores for all samples using the Loops That Matter method
        """
        if int_var == None:
            int_var = list(self.intervention_effects.keys())[0]  # Select an intervention for which we will assess the feedback loop dominance
        j = self.intervention_variables.index(int_var)

        ## Get the right loop scores for specific intervention
        linkscores_per_sample = []
        loopscores_per_sample = []
        loopscores_combined_per_sample = []
        for k in range(self.N):  # Loop over all samples
            df_i = self.df_sol_per_sample[k][j]
            params = {target : {source : self.param_samples[int_var][target][source][k]
                                        for source in self.param_samples[int_var][target]}
                                        for target in self.param_samples[int_var]}
            linkscores = self.get_link_scores(df_i, params)
            loopscores, feedback_loops = self.get_loop_scores(linkscores)
            linkscores_per_sample += [linkscores]
            loopscores_per_sample += [loopscores]
            loopscores_combined_over_time = {loop : np.mean(loopscores[loop]) for loop in loopscores}
            loopscores_combined_per_sample += [loopscores_combined_over_time]

        loopscores_combined_per_sample = {loop : [loopscores_combined_per_sample[k][loop] for k in range(self.N)] for loop in loopscores_combined_per_sample[0]}

        df_loops = pd.DataFrame(loopscores_combined_per_sample)
        df_loops = df_loops.reindex(columns=list(
                            df_loops.abs().median().sort_values(ascending=False).index))
        self.df_loops = df_loops
        self.loopscores_per_sample = loopscores_per_sample
        return df_loops, loopscores_per_sample

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
        # elif self.prior == "halfnormal":
        #     sample_pars = halfnorm.rvs(loc = 0, scale = self.parameter_value, size=(num_pars))
        #     sample_pars_int = halfnorm.rvs(loc = 0, scale = self.parameter_value/2, size=(num_pars_int))

        # # Draw samples from the prior distribution
        # if self.prior == "uniform":
        #     sample_pars_aux = np.random.uniform(0, self.parameter_value_aux, size=(par_count_aux, self.N))
        #     sample_pars_stocks = np.random.uniform(0, self.parameter_value_stocks, size=(par_count_stocks, self.N))
        # elif self.prior == "halfnormal":
        #     sample_pars_aux = halfnorm.rvs(loc = 0, scale = self.parameter_value_aux, size=(par_count_aux, self.N))
        #     sample_pars_stocks = halfnorm.rvs(loc = 0, scale = self.parameter_value_stocks, size=(par_count_stocks, self.N))

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
                #if self.df_adj.loc[var_2, var] != 0:
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
        

                        #   params[var][var_2 + " * " + var_3] = (params[var][var_2 + " * " + var_3] * 2) - self.parameter_value/2  # [-self.max_parameter_value/2, self.max_parameter_value/2]
   
        self.params = params
        return params

    def make_equations_auxiliary_independent(self, params):
        """" Create independent equations without auxiliaries.
        Input: parameter dictionary with auxiliary terms
        Output: parameter dictionary without auxiliary terms (i.e., only in terms of stocks and constants)
        """
        self.params = params
        new_params = deepcopy(self.params)

        original_equations = {var : " + ".join([pred + " * " + str(new_params[var][pred]) if
                                                pred != "Intercept" else
                                                str(new_params[var][pred])
                                                for pred in new_params[var]]) for var in self.stocks_and_auxiliaries}

        new_equations = original_equations

        # print("Original equations: ", original_equations)

        def contains_aux(eq, auxiliaries):
            for aux in auxiliaries:
                pattern = r'\b' + re.escape(aux) + r'\b'
                if re.search(pattern, eq):
                    return True
            return False
        
        for k, var in enumerate(self.auxiliaries + self.stocks):  # Iterate over all variables, starting with auxiliaries
            eq = original_equations[var]
            # print("Eq before change: ", eq)
            if np.any([aux in eq for aux in self.auxiliaries]):  # If the equation contains auxiliaries
                count = 0 
                while contains_aux(eq, self.auxiliaries):
                #np.any([aux in eq for aux in self.auxiliaries]):  # Iterate until all auxiliaries are removed from the equations
                    for aux in sorted(self.auxiliaries, key=len, reverse=True):  # sort longest names first!
                        pattern = r'\b' + re.escape(aux) + r'\b'  # match whole words only
                        eq = re.sub(pattern, f"( {new_equations[aux]} )", eq)
                        
                        #eq = eq.replace(aux, "( " + aux + " )")
                        #eq = eq.replace(aux, new_equations[aux])
                        #print(eq)
                    eq_sym = sym.simplify(parse_expr(eq))
                    eq = str(eq_sym)
                    
                    count += 1

                    if count == 20: ### Temporary warning to prevent infinite loops; will be removed later by informative warning regarding feedback between auxiliaries
                        print("Was unable to get rid of the auxiliaries in eq for var: ", var, " = ", eq_sym, "\n")
                        break

                eq_sym_exp = sym.expand(eq_sym)  # Expand the equation to get rid of parentheses

                new_params[var] = {str(key) : float(eq_sym_exp.as_coefficients_dict()[key]) for key in eq_sym_exp.as_coefficients_dict()}

                if "1" in new_params[var]:  # If an intercept term is present
                    new_params[var]["Intercept"] = new_params[var]["1"]  # Rename the "1" key to "Intercept"
                    new_params[var] = {key : new_params[var][key] for key in new_params[var] if key != "1"}  # Remove the "1" key

       # new_params = {var : new_params[var] for var in self.stocks_and_auxiliaries} 
        self.new_params = new_params
        return new_params

    def get_A_and_K_matrices(self):
        """ Create matrices A and K from the parameter dictionary without auxiliary terms.
            Also returns the intercept terms as a vector.
        """
        params_wo_auxiliaries = {var : self.new_params[var] for var in self.stocks}  # Remove auxiliaries from the parameter dictionary

        A = np.zeros((len(self.stocks_and_constants), len(self.stocks_and_constants)), order='F')
        K = np.zeros((len(self.stocks_and_constants), len(self.stocks_and_constants), len(self.stocks_and_constants)), order='F')
        b = np.zeros(len(self.stocks_and_constants), order='F')

        for i, var in enumerate(self.stocks_and_constants):
            if var in self.stocks and "Intercept" in params_wo_auxiliaries[var]:
                b[i] = params_wo_auxiliaries[var]["Intercept"]
            else:  # Constants; no intercept term
                b[i] = 0

        for destination in params_wo_auxiliaries:
            for origin in params_wo_auxiliaries[destination]:
                if origin != "Intercept":
                    if "*" in origin:  # Interaction term
                        destination_index = self.stocks_and_constants.index(destination)
                        if "**" in origin:
                            origin_1_index = self.stocks_and_constants.index(origin.split("**")[0])
                            origin_2_index = self.stocks_and_constants.index(origin.split("**")[0])
                        else:
                            origin_split = origin.split("*")
                            origin_1_index = self.stocks_and_constants.index(origin_split[0])
                            origin_2_index = self.stocks_and_constants.index(origin_split[1])
                        K[destination_index, origin_2_index, origin_1_index] = params_wo_auxiliaries[destination][origin]

                    else:  # Not an interaction term
                        destination_index = self.stocks_and_constants.index(destination)
                        origin_index = self.stocks_and_constants.index(origin)
                        A[destination_index, origin_index] = params_wo_auxiliaries[destination][origin]
        return A, K, b

    def run_SDM(self, x0, A, K, b):
        """ Run the SDM and return a dataframe with all the variables at every time step, including auxiliaries.
        """
        if self.interaction_terms:
            solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, jac=self.jac,
                                method=self.solver, rtol=1e-6, atol=1e-6).y
        else:  # Linear system
            if self.solve_analytically: 
                solution = self.analytical_solution(self.t_eval[:, None], x0, A, b).T
            else:
                solution = solve_ivp(self.solve_sdm_linear, self.t_span, x0, args=(A, b),
                                   t_eval=self.t_eval, jac=self.jac_linear,
                                   method=self.solver, rtol=1e-6, atol=1e-6).y

        if np.sum(solution > 10):
            print("Warning: Solution has values larger than 10. The maximum parameter values may be too large.")
        #if np.sum(solution > 1) > 0 == False:
        #    print("Warning: Solution does not have values larger than 1. The maximum parameter values may be too small.")
        df_sol = pd.DataFrame(solution.T, columns=self.stocks_and_constants, index=self.t_eval)
        df_sol["Time"] = df_sol.index

        params_wo_stocks = {var : self.new_params[var] for var in self.auxiliaries}
        df_sol_with_aux = self.evaluate_auxiliaries(params_wo_stocks, df_sol, self.t_eval)
        return df_sol_with_aux

    def analytical_solution(self, t, x0, A, b):
        """ Analytical solution for a linear system of ODEs.
            We use the Pseudo-inverse because the regular inverse only works for non-singular matrices.
        """
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
           print("Matrix A is singular. Using the pseudo-inverse instead.")
           A_inv = np.linalg.pinv(A)  # Pseudo-inverse for singular matrices
        I = np.identity(A.shape[0])
        A_inv_b = np.matmul(A_inv, b)
        sol = np.zeros((self.t_eval.shape[0], x0.shape[0]))
        sol[0, :] = x0  # Initial condition
        for i, t in enumerate(self.t_eval[1:]):  # Skip the first time point, which is x0
            exp_At = scipy.linalg.expm(A * t)
            sol[i, :] = np.matmul((exp_At - I), A_inv_b) + np.matmul(exp_At, x0)
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

    def evaluate_auxiliaries(self, params, df_sol, t_eval):
        """ Evaluate the auxiliary variables at each time step.
        Input: parameter dictionary with auxiliary terms, and the solution dataframe
        Output: list of auxiliary values at each time step
        """
        df_sol_with_auxiliaries = df_sol.copy()
        for aux in self.auxiliaries:
            aux_values = []
            for t in t_eval:
                aux_value = 0
                for origin in params[aux]:
                    if origin == "Intercept":
                        aux_value += params[aux][origin]
                    else:
                        if "*" in origin:  # Interaction term
                            origin_1 = origin.split("*")[0]
                            origin_2 = origin.split("*")[1]
                            aux_value += params[aux][origin] * df_sol[origin_1][int(t)] * df_sol[origin_2][int(t)]
                        else:  # Not an interaction term
                            aux_value += params[aux][origin] * df_sol[origin][int(t)]
                aux_values.append(aux_value)
            df_sol_with_auxiliaries[aux] = aux_values
        return df_sol_with_auxiliaries

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
        
### TESTING ###
    def f_no_aux(self, time, x, params_wo_auxiliaries):
        """ Test the vectorized equation x' = Ax + Kxx.
        """
        eqs = []
        for var in self.stocks_and_constants:
            eq_i = 0

            if var not in self.constants:
                for pred in params_wo_auxiliaries[var]:
                    if pred == "Intercept":
                        eq_i += params_wo_auxiliaries[var][pred]
                    else:
                        if "**" in pred:  # Quadratic term
                            eq_i += x[self.stocks_and_constants.index(pred.split("*")[0])]**2 * params_wo_auxiliaries[var][pred]
                        elif "*" in pred:  # Interaction term
                            eq_i += (x[self.stocks_and_constants.index(pred.split("*")[0])] * 
                                     x[self.stocks_and_constants.index(pred.split("*")[1])] * 
                                     params_wo_auxiliaries[var][pred])
                        else:  # Linear term
                            eq_i += x[self.stocks_and_constants.index(pred)] * params_wo_auxiliaries[var][pred]
            eqs += [eq_i]
        
        return np.array(eqs)

    def test_vectorized_eqs(self):
        """ Test whether the vectorized equations are the same as the non-vectorized equations.
        """
        self.params = self.sample_model_parameters()  #([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent(self.params)  # Remove auxiliaries from the equations
        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries

        # Obtain the vectorized solution
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                            t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        
        # Obtain the test solution
        params_wo_auxiliaries = {var : self.new_params[var] for var in self.stocks}  # Remove auxiliaries from the parameter dictionary
        sol_test = solve_ivp(self.f_no_aux, self.t_span, x0, args=(params_wo_auxiliaries,), 
                             t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        #df_sol_test = pd.DataFrame(sol_test.y.T, columns=s.stocks_and_constants, index=t_eval)

        #assert ((df_sol_per_sample[-1][-1]-df_sol_test)**2).sum().sum() < 1e-15 # Check if the solutions are the same
        assert np.allclose(sol_test.y, solution.y)
        print("Test comparison with vectorized implementation passed.")

    def f_manual(self, time, x, params):
        """ Manual equations for the Sleep example for testing purposes.
        """
        # Auxiliaries
        p_a = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Physical_activity"]["Depressive_symptoms"] + 
                params["Physical_activity"]["Intercept"])# Physical activity
        p_p = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Proinflammatory_processes"]["Depressive_symptoms"] + 
                p_a * params["Proinflammatory_processes"]["Physical_activity"] +
                x[self.stocks_and_constants.index("Body_fat")] * params["Proinflammatory_processes"]["Body_fat"] +
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Proinflammatory_processes"]["Perceived_stress"] +
                params["Proinflammatory_processes"]["Intercept"]) # Proinflammatory processes
        s_p = (x[self.stocks_and_constants.index("Body_fat")] * params["Sleep_problems"]["Body_fat"] + 
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Sleep_problems"]["Perceived_stress"] +
                p_p * params["Sleep_problems"]["Proinflammatory_processes"] + params["Sleep_problems"]["Intercept"]) # Sleep problems
        t_m = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Treatment"]["Depressive_symptoms"] + 
                params["Treatment"]["Intercept"])
        # print("Auxiliaries: ", p_a, p_p, s_p, t_m)

        # Stocks, order: Depressive_symptoms, Childhood_adversity, Body_fat, Perceived_stress, Treatment
        d_s = (s_p * params["Depressive_symptoms"]["Sleep_problems"] + 
                x[self.stocks_and_constants.index("Childhood_adversity")] * params["Depressive_symptoms"]["Childhood_adversity"] +
                x[self.stocks_and_constants.index("Perceived_stress")] * params["Depressive_symptoms"]["Perceived_stress"] +
                #x[self.stocks_and_constants.index("Treatment")] * params["Depressive_symptoms"]["Treatment"] + 
                t_m * params["Depressive_symptoms"]["Treatment"] + 
               p_p * params["Depressive_symptoms"]["Proinflammatory_processes"] + params["Depressive_symptoms"]["Intercept"])  # Depressive symptoms
        if self.interaction_terms:
            d_s += s_p * x[self.stocks_and_constants.index("Perceived_stress")] * params["Depressive_symptoms"]["Perceived_stress * Sleep_problems"]
        c_a = 0  # Childhod adversity
        b_f = (s_p * params["Body_fat"]["Sleep_problems"] + 
                p_a * params["Body_fat"]["Physical_activity"] + params["Body_fat"]["Intercept"])  # Body fat
        # if self.interaction_terms:
        #     b_f += s_p * p_a * params["Body_fat"]["Physical_activity * Sleep_problems"]
        p_s = (s_p * params["Perceived_stress"]["Sleep_problems"] +
                x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Perceived_stress"]["Depressive_symptoms"] +
                x[self.stocks_and_constants.index("Childhood_adversity")] * params["Perceived_stress"]["Childhood_adversity"] + 
                params["Perceived_stress"]["Intercept"])  # Perceived stress'
        #t_m = (x[self.stocks_and_constants.index("Depressive_symptoms")] * params["Treatment"]["Depressive_symptoms"] + 
        #        params["Treatment"]["Intercept"]) # Treatment
        return np.array([d_s, c_a, b_f, p_s])#, t_m])

    def test_with_sleep_depression_model(self):
        """ Test whether the vectorized equations are the same as the manually implemented equations.
        """
        self.params = self.sample_model_parameters()  #([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent(self.params)  # Remove auxiliaries from the equations
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries
        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        
        sol_test = solve_ivp(self.f_manual, self.t_span, x0, args=(self.params,), 
                                t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6)
        assert np.allclose(sol_test.y, solution.y)
        print("Test comparison with manual implementation for Sleep example passed.")

    #### Compare the results to straightforward implementation of equations
    def test_with_linear_model(self):
        """ Test whether the algebraic solution is similar to the numerical solution.
        """
        self.params = self.sample_model_parameters()  #([])  # Sample model parameters
        self.new_params = self.make_equations_auxiliary_independent(self.params)  # Remove auxiliaries from the equations
        A, K, b = self.get_A_and_K_matrices()  # Get A and K matrices and intercept vector from the parameter dictionary without auxiliaries

        x0 = np.ones(len(self.stocks_and_constants), order='F') * 0.01  
        solution = solve_ivp(self.solve_sdm, self.t_span, x0, args=(A, K, b),
                                t_eval=self.t_eval, method=self.solver, rtol=1e-12, atol=1e-12)
        analytical_solution = self.analytical_solution(self.t_eval[:, None], x0, A, b).T
        assert np.allclose(analytical_solution, solution.y)
        print("Test comparison analytic and numerical solution for linear model passed.")

    def test_get_link_scores(self):
        """ Test the get_link_scores function with the test data from the Loops that matter paper (Table 1)
        https://onlinelibrary.wiley.com/doi/full/10.1002/sdr.1658

        Table 1:
        variable / time 1 / time 2 / variable change / partial change in z / link score magnitude
        x / 5 / 7 / 2 / 4 / (4/5)
        y / 4 / 5 / 1 / 1 / (1/5)
        z / 14 / 19 / 5 / - / -
        """
        # Define the parameters for the test based on Table 1
        test_params = {
            "Z": {"X": 2, "Y": 1},
            "X": {"b_x": 2},
            "Y": {"b_y": 1}
        }
        # Store original values
        original_t_eval = self.t_eval
        original_stocks_and_auxiliaries = self.stocks_and_auxiliaries
        original_stocks = self.stocks

        self.t_eval = [1, 2]  # Time steps
        self.stocks_and_auxiliaries = ["X", "Y", "Z"]
        self.stocks = []

        # Create the DataFrame for the test
        df_test_loops = pd.DataFrame({
            "X": [5, 7],
            "Y": [4, 5],
            "b_x" : [2, 2],
            "b_y" : [1, 1],
            "Z": [14, 19]
        }, index=self.t_eval)

        # Call the get_link_scores function with the test data
        link_scores = self.get_link_scores(df_test_loops, test_params)

        # Verify the results
        # print(link_scores)
        assert link_scores[self.t_eval[-1]]["Z"]["X"] == 4/5
        assert link_scores[self.t_eval[-1]]["Z"]["Y"] == 1/5

        # Return original values
        self.t_eval = original_t_eval
        self.stocks_and_auxiliaries = original_stocks_and_auxiliaries
        self.stocks = original_stocks

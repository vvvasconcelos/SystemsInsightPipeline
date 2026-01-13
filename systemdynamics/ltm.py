"""
Loops That Matter (LTM) - Pathway-based loop dominance analysis for System Dynamics models.

Based on the theory described by Schoenberg, Davidsen and Eberlein (2019):
https://arxiv.org/abs/1908.11434

This module implements quantitative loop dominance analysis following the LTM method:

1. **Link scores** (Equation 1 in paper): For non-integration links,
   - Δz_x = ceteris paribus change in z due to x changing (other inputs held at t-Δt values)
   - link_score = sign(Δz_x/Δx) * |Δz_x/Δz|
   - Magnitude can exceed 1 when there are cancellations among inputs

2. **Flow→Stock link scores** (Equation 2): For integration links,
   - score = flow_value / net_flow (positive for inflows, negative for outflows)
   
3. **Loop scores**: Product of link scores around feedback loops (can exceed 1)

4. **Relative loop scores**: Normalized by sum of absolute loop scores (bounded to [-1, 1])

The output is a pandas DataFrame suitable for analysis and visualization of loop dominance over time.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import solve_ivp
from copy import deepcopy
from tqdm import tqdm


class LoopsThatMatter:
    """
    Loops That Matter (LTM) analysis for System Dynamics models.
    
    Implements the pathway-based loop dominance method from Schoenberg, Davidsen & Eberlein (2019).
    
    Key distinction from simpler approaches:
    - Link scores use ceteris paribus partial changes, not simple derivatives
    - Link scores can have magnitude > 1 (indicating amplification or cancellation effects)
    - Flow→stock links use special integration-based scoring
    - Loop scores are products of link scores, normalized only at the final step
    
    Attributes:
        sdm: Reference to the SDM model instance
        feedback_loops: List of feedback loops (each is a list of variable names)
        loop_names: List of loop names (e.g., "R1", "B1", etc.)
        ltm_link_scores_df: DataFrame of proper LTM link scores (can be > 1)
        relative_link_scores_df: DataFrame of visualization-friendly scores (bounded [-1,1])
    """
    
    def __init__(self, sdm):
        """
        Initialize LTM analysis for an SDM model.
        
        Args:
            sdm: An SDM instance with model structure (adjacency matrix, variables, etc.)
        """
        self.sdm = sdm
        self.df_adj = sdm.df_adj
        self.variables = sdm.variables
        self.stocks = sdm.stocks
        self.auxiliaries = sdm.auxiliaries
        self.constants = sdm.constants
        
        # Find all feedback loops in the model
        self.feedback_loops = self._find_feedback_loops()
        self.loop_names = self._name_loops()
        
        # Build dependency graph for equation evaluation
        self._build_dependency_info()
        
        # Store results - distinguish LTM scores from visualization scores
        self.ltm_link_scores_df = None      # True LTM scores (Eq 1, can be > 1)
        self.relative_link_scores_df = None  # For visualization (bounded [-1,1])
        self.loop_scores_df = None
        
        # Legacy alias
        self.link_scores_df = None
        
    def _build_dependency_info(self):
        """
        Build information about variable dependencies for ceteris paribus evaluation.
        
        For each variable, stores:
        - inputs: list of variables that directly affect it
        - is_stock: whether it's a stock (integrated) variable
        - is_flow_target: for stocks, the "flow" is the rate of change
        """
        self.var_inputs = {}
        self.var_is_stock = {}
        
        for var in self.variables:
            # Find all inputs to this variable from adjacency matrix
            inputs = []
            for source in self.df_adj.columns:
                if self.df_adj.loc[var, source] != 0:
                    inputs.append(source)
            self.var_inputs[var] = inputs
            self.var_is_stock[var] = var in self.stocks
        
    def _find_feedback_loops(self, max_length=None):
        """
        Find all feedback loops in the model using the adjacency matrix.
        
        Args:
            max_length: Maximum loop length to search for. If None, uses number of stocks + auxiliaries.
        
        Returns:
            List of loops, where each loop is a list of variable names in order.
        """
        if max_length is None:
            max_length = len(self.stocks) + len(self.auxiliaries)
        
        # Create directed graph from adjacency matrix
        # Note: df_adj[i,j] != 0 means j influences i, so we need to transpose for nx
        adj_matrix = np.array(self.df_adj).T
        adj_binary = (adj_matrix != 0).astype(int)
        
        G = nx.from_numpy_array(adj_binary, create_using=nx.DiGraph)
        var_names = list(self.df_adj.columns)
        G = nx.relabel_nodes(G, dict(enumerate(var_names)))
        
        # Find all simple cycles up to max_length
        feedback_loops = list(nx.simple_cycles(G, length_bound=max_length))
        
        # Filter to only include loops that contain at least one stock
        loops_with_stocks = []
        for loop in feedback_loops:
            if any(var in self.stocks for var in loop):
                loops_with_stocks.append(loop)
        
        return loops_with_stocks
    
    def _name_loops(self):
        """
        Name feedback loops as R1, R2, ... (reinforcing) or B1, B2, ... (balancing).
        
        Returns:
            List of loop names corresponding to self.feedback_loops
        """
        loop_names = []
        r_count = 0
        b_count = 0
        
        for loop in self.feedback_loops:
            polarity = self._get_loop_polarity(loop)
            if polarity > 0:
                r_count += 1
                loop_names.append(f"R{r_count}")
            else:
                b_count += 1
                loop_names.append(f"B{b_count}")
        
        return loop_names
    
    def _get_loop_polarity(self, loop):
        """
        Determine the polarity of a feedback loop.
        
        A loop is reinforcing (+) if the product of all link polarities is positive,
        and balancing (-) if the product is negative.
        
        Args:
            loop: List of variable names forming a loop
        
        Returns:
            +1 for reinforcing loop, -1 for balancing loop
        """
        polarity = 1
        n = len(loop)
        
        for i in range(n):
            source = loop[i]
            target = loop[(i + 1) % n]  # Wrap around to close the loop
            
            # Get the link weight from adjacency matrix
            # df_adj[target, source] gives the influence of source on target
            link_weight = self.df_adj.loc[target, source]
            
            if link_weight == -999:
                # Unknown polarity, assume positive
                link_weight = 1
            
            if link_weight < 0:
                polarity *= -1
        
        return polarity
    
    def _get_link_coefficient(self, source, target, params):
        """
        Get the coefficient for a link from source to target.
        
        Args:
            source: Source variable name
            target: Target variable name
            params: Parameter dictionary
        
        Returns:
            The coefficient (partial derivative) for the link
        """
        if target in params and source in params[target]:
            return params[target][source]
        return 0.0
    
    def _evaluate_target_ceteris_paribus(self, target, source, var_values_curr, var_values_prev, params):
        """
        Evaluate target with source at current value, all other inputs at previous values.
        
        This computes the ceteris paribus change Δz_x from LTM Equation 1.
        
        For a linear model z = Σ(coef_i * x_i), this is:
            z_ceteris_paribus = coef_source * source_curr + Σ(coef_i * x_i_prev) for i != source
        
        Args:
            target: Target variable name
            source: Source variable name (the one that changes)
            var_values_curr: Dict of variable values at current time
            var_values_prev: Dict of variable values at previous time
            params: Parameter dictionary
        
        Returns:
            The value of target if only source had changed
        """
        result = 0.0
        
        # Get all inputs to target
        for input_var in self.var_inputs.get(target, []):
            coef = self._get_link_coefficient(input_var, target, params)
            
            if input_var == source:
                # Use current value for the source
                result += coef * var_values_curr.get(input_var, 0.0)
            else:
                # Use previous value for all other inputs (ceteris paribus)
                result += coef * var_values_prev.get(input_var, 0.0)
        
        return result
    
    def _compute_net_flow_for_stock(self, stock, var_values, params):
        """
        Compute the net flow into a stock variable.
        
        For stocks, the "net flow" is dx/dt = Σ(coef_i * x_i) for all inputs.
        
        Args:
            stock: Stock variable name
            var_values: Dict of current variable values
            params: Parameter dictionary
        
        Returns:
            Net flow value
        """
        net_flow = 0.0
        for input_var in self.var_inputs.get(stock, []):
            coef = self._get_link_coefficient(input_var, stock, params)
            net_flow += coef * var_values.get(input_var, 0.0)
        return net_flow
    
    def _compute_flow_contribution(self, source, stock, var_values, params):
        """
        Compute the flow contribution from source to stock.
        
        This is coef * source_value, which contributes to the stock's rate of change.
        
        Args:
            source: Source variable name
            stock: Stock variable name
            var_values: Dict of current variable values
            params: Parameter dictionary
        
        Returns:
            Flow contribution value
        """
        coef = self._get_link_coefficient(source, stock, params)
        return coef * var_values.get(source, 0.0)
    
    def compute_link_scores(self, df_sol, params, t_eval):
        """
        Compute LTM link scores for all links at each time step.
        
        Implements Equation 1 (non-integration links) and Equation 2 (flow→stock links)
        from Schoenberg, Davidsen & Eberlein (2019).
        
        For non-integration links (Equation 1):
            Δz_x = ceteris paribus change = f(x_t, y_{t-1}, ...) - f(x_{t-1}, y_{t-1}, ...)
            link_score = sign(Δz_x/Δx) * |Δz_x/Δz|
            
        For flow→stock links (Equation 2):
            link_score = flow_contribution / net_flow
        
        Args:
            df_sol: Solution dataframe with variable values at each time step
            params: Parameter dictionary
            t_eval: Array of time points
        
        Returns:
            DataFrame with LTM link scores (can have magnitude > 1)
        """
        ltm_scores_data = []      # True LTM scores
        relative_scores_data = [] # Visualization scores (bounded)
        
        # Get all active links from the adjacency matrix
        active_links = []
        for target in self.df_adj.index:
            for source in self.df_adj.columns:
                if self.df_adj.loc[target, source] != 0:
                    active_links.append((source, target))
        
        n_times = len(t_eval)
        
        for t_idx in range(1, n_times):
            t = t_eval[t_idx]
            t_prev = t_eval[t_idx - 1]
            
            # Build value dictionaries for current and previous time
            var_values_curr = {}
            var_values_prev = {}
            for var in self.variables:
                if var in df_sol.columns:
                    var_values_curr[var] = df_sol.loc[t, var]
                    var_values_prev[var] = df_sol.loc[t_prev, var]
            
            # Compute LTM link scores for each link
            time_ltm_scores = {}
            time_relative_scores = {}
            
            for source, target in active_links:
                is_stock_target = target in self.stocks
                
                # Get actual changes
                delta_source = var_values_curr.get(source, 0.0) - var_values_prev.get(source, 0.0)
                delta_target = var_values_curr.get(target, 0.0) - var_values_prev.get(target, 0.0)
                
                if is_stock_target:
                    # EQUATION 2: Flow→Stock link score
                    # score = flow_contribution / net_flow
                    net_flow_curr = self._compute_net_flow_for_stock(target, var_values_curr, params)
                    flow_contrib = self._compute_flow_contribution(source, target, var_values_curr, params)
                    
                    if abs(net_flow_curr) > 1e-10:
                        ltm_score = flow_contrib / net_flow_curr
                    else:
                        ltm_score = 0.0
                        
                else:
                    # EQUATION 1: Non-integration link score
                    # Δz_x = f(x_t, others_{t-1}) - f(x_{t-1}, others_{t-1})
                    # link_score = sign(Δz_x/Δx) * |Δz_x/Δz|
                    
                    if abs(delta_source) < 1e-10 or abs(delta_target) < 1e-10:
                        ltm_score = 0.0
                    else:
                        # Compute ceteris paribus value (source at t, others at t-1)
                        z_ceteris = self._evaluate_target_ceteris_paribus(
                            target, source, var_values_curr, var_values_prev, params
                        )
                        # Compute baseline (all at t-1)
                        z_baseline = var_values_prev.get(target, 0.0)
                        
                        # Δz_x = ceteris paribus change
                        delta_z_x = z_ceteris - z_baseline
                        
                        # LTM score: sign(Δz_x/Δx) * |Δz_x/Δz|
                        if abs(delta_z_x) < 1e-10:
                            ltm_score = 0.0
                        else:
                            polarity = np.sign(delta_z_x / delta_source)
                            magnitude = abs(delta_z_x / delta_target)
                            ltm_score = polarity * magnitude
                
                time_ltm_scores[(source, target)] = ltm_score
            
            # Compute relative scores for visualization (normalize per target)
            for target in set(t for s, t in active_links):
                target_links = [(s, t) for s, t in active_links if t == target]
                total_abs = sum(abs(time_ltm_scores.get(link, 0.0)) for link in target_links)
                
                for link in target_links:
                    ltm_score = time_ltm_scores.get(link, 0.0)
                    if total_abs > 1e-10:
                        relative_score = ltm_score / total_abs
                    else:
                        relative_score = 0.0
                    time_relative_scores[link] = relative_score
            
            # Store scores
            for source, target in active_links:
                delta_source = var_values_curr.get(source, 0.0) - var_values_prev.get(source, 0.0)
                delta_target = var_values_curr.get(target, 0.0) - var_values_prev.get(target, 0.0)
                
                ltm_scores_data.append({
                    'time': t,
                    'source': source,
                    'target': target,
                    'ltm_link_score': time_ltm_scores.get((source, target), 0.0),
                    'relative_link_score': time_relative_scores.get((source, target), 0.0),
                    'delta_source': delta_source,
                    'delta_target': delta_target,
                    'is_stock_target': target in self.stocks
                })
        
        self.ltm_link_scores_df = pd.DataFrame(ltm_scores_data)
        
        # Create legacy-compatible link_scores_df with relative scores
        # (for backward compatibility with existing visualization code)
        legacy_data = []
        for _, row in self.ltm_link_scores_df.iterrows():
            legacy_data.append({
                'time': row['time'],
                'source': row['source'],
                'target': row['target'],
                'link_score': row['relative_link_score'],  # Use relative for visualization
                'ltm_score': row['ltm_link_score'],        # Keep LTM score available
                'raw_contribution': row['ltm_link_score'] * row['delta_target'] if row['delta_target'] != 0 else 0,
                'delta_target': row['delta_target']
            })
        self.link_scores_df = pd.DataFrame(legacy_data)
        
        return self.ltm_link_scores_df
    
    def compute_loop_scores(self, link_scores_df=None, use_ltm_scores=True):
        """
        Compute loop scores by multiplying link scores around each feedback loop.
        
        The loop score is the product of LTM link scores for all links in the loop.
        Since LTM link scores can exceed 1, loop scores can also exceed 1.
        
        Relative loop scores are normalized by sum of absolute loop scores.
        
        Args:
            link_scores_df: DataFrame of link scores (uses self.ltm_link_scores_df if None)
            use_ltm_scores: If True, use LTM scores (can be > 1). If False, use relative scores.
        
        Returns:
            DataFrame with loop scores indexed by time, one column per loop
        """
        if link_scores_df is None:
            link_scores_df = self.ltm_link_scores_df
        
        if link_scores_df is None:
            raise ValueError("No link scores available. Call compute_link_scores first.")
        
        # Determine which score column to use
        score_col = 'ltm_link_score' if use_ltm_scores else 'relative_link_score'
        if score_col not in link_scores_df.columns:
            score_col = 'link_score'  # Fallback for legacy format
        
        times = link_scores_df['time'].unique()
        
        loop_scores_data = []
        
        for t in times:
            t_data = link_scores_df[link_scores_df['time'] == t]
            
            # Create lookup for link scores at this time
            link_lookup = {}
            for _, row in t_data.iterrows():
                link_lookup[(row['source'], row['target'])] = row[score_col]
            
            # Compute score for each loop
            loop_raw_scores = {}
            loop_missing_links = {}
            
            for loop_idx, loop in enumerate(self.feedback_loops):
                loop_name = self.loop_names[loop_idx]
                n = len(loop)
                
                # Product of link scores around the loop
                loop_score = 1.0
                missing_links = []
                
                for i in range(n):
                    source = loop[i]
                    target = loop[(i + 1) % n]
                    link = (source, target)
                    
                    if link in link_lookup:
                        link_score = link_lookup[link]
                        # Note: We multiply even if score is 0 (loop becomes 0)
                        loop_score *= link_score
                    else:
                        # Link not computed - this is a problem for LTM
                        # Record it and set loop score to NaN (unknown)
                        missing_links.append(link)
                
                if missing_links:
                    # Cannot compute valid LTM score with missing links
                    loop_raw_scores[loop_name] = np.nan
                    loop_missing_links[loop_name] = missing_links
                else:
                    loop_raw_scores[loop_name] = loop_score
            
            # Normalize loop scores (only among non-NaN scores)
            valid_scores = {k: v for k, v in loop_raw_scores.items() if not np.isnan(v)}
            total_abs = sum(abs(s) for s in valid_scores.values())
            
            row_data = {'time': t}
            for loop_name, score in loop_raw_scores.items():
                row_data[f'{loop_name}_raw'] = score
                
                if np.isnan(score):
                    row_data[f'{loop_name}_relative'] = np.nan
                elif total_abs > 1e-10:
                    row_data[f'{loop_name}_relative'] = score / total_abs
                else:
                    row_data[f'{loop_name}_relative'] = 0.0
            
            loop_scores_data.append(row_data)
        
        self.loop_scores_df = pd.DataFrame(loop_scores_data)
        return self.loop_scores_df
    
    def run_ltm_analysis(self, params, t_start=0, t_end=None, n_points=100, 
                         intervention_intensities=None, intervention_variable=None,
                         progress_bar=True):
        """
        Run a complete LTM analysis: simulate the model and compute all scores.
        
        Args:
            params: Parameter dictionary for the model
            t_start: Start time for simulation
            t_end: End time for simulation (uses model default if None)
            n_points: Number of time points to evaluate
            intervention_intensities: Array of intensities for each intervention variable.
                                     If None, no intervention is applied.
                                     Can also be a single float if intervention_variable is specified.
            intervention_variable: If specified with a single intensity, apply intervention
                                  to this variable only (others get 0).
            progress_bar: Whether to show progress bar
        
        Returns:
            Tuple of (solution_df, link_scores_df, loop_scores_df)
        """
        if t_end is None:
            t_end = self.sdm.t_span[1]
        
        t_eval = np.linspace(t_start, t_end, n_points)
        
        # Store original t_eval and t_span
        original_t_eval = self.sdm.t_eval
        original_t_span = self.sdm.t_span
        
        # Set new t_eval for detailed time points
        self.sdm.t_eval = t_eval
        self.sdm.t_span = [t_start, t_end]
        
        try:
            # Handle intervention intensities
            if intervention_intensities is not None:
                if intervention_variable is not None:
                    # Single variable intervention
                    intensities = np.zeros(len(self.sdm.intervention_variables))
                    var_idx = self.sdm.intervention_variables.index(intervention_variable)
                    intensities[var_idx] = float(intervention_intensities)
                    intervention_intensities = intensities
                else:
                    # Convert to numpy array and ensure float type
                    intervention_intensities = np.asarray(intervention_intensities, dtype=np.float64)
                    # Replace any NaN with 0
                    intervention_intensities = np.nan_to_num(intervention_intensities, nan=0.0)
                
                # Use the SDM method to run with interventions
                df_sol = self.sdm.run_SDM_with_intervention_intensities(intervention_intensities, params)
            else:
                # No intervention - run baseline
                x0 = np.zeros(len(self.stocks), dtype=np.float64)
                constants_values = np.zeros(len(self.constants), dtype=np.float64)
                
                if self.sdm.interaction_terms:
                    A, b = None, None
                else:
                    A, b = self.sdm.params_to_A_b(params, constants_values)
                
                df_sol = self.sdm.run_SDM(x0, constants_values, A, b, params)
        finally:
            # Restore original t_eval and t_span
            self.sdm.t_eval = original_t_eval
            self.sdm.t_span = original_t_span
        
        # Compute link and loop scores
        link_scores_df = self.compute_link_scores(df_sol, params, t_eval)
        loop_scores_df = self.compute_loop_scores()
        
        return df_sol, link_scores_df, loop_scores_df
    
    def run_ltm_across_samples(self, n_samples=None, t_start=0, t_end=None, n_points=50,
                                intervention_intensities=None, intervention_variable=None):
        """
        Run LTM analysis across multiple parameter samples and aggregate results.
        
        This pools the loop scores across different parameter realizations to
        understand robust patterns of loop dominance.
        
        Args:
            n_samples: Number of parameter samples (uses model N if None)
            t_start: Start time for simulation
            t_end: End time for simulation
            n_points: Number of time points per simulation
            intervention_intensities: Array of intensities for each intervention variable.
                                     If None, no intervention is applied.
            intervention_variable: If specified with a single intensity, apply intervention
                                  to this variable only.
        
        Returns:
            DataFrame with aggregated loop scores (mean, std, etc.) across samples
        """
        if n_samples is None:
            n_samples = self.sdm.N
        
        all_loop_scores = []
        
        for sample_idx in tqdm(range(n_samples), desc="Running LTM across samples"):
            # Sample parameters
            params = self.sdm.sample_model_parameters()
            
            try:
                # Run LTM analysis with intervention
                _, _, loop_scores_df = self.run_ltm_analysis(
                    params, t_start, t_end, n_points, 
                    intervention_intensities=intervention_intensities,
                    intervention_variable=intervention_variable,
                    progress_bar=False
                )
                
                # Add sample index
                loop_scores_df['sample_idx'] = sample_idx
                all_loop_scores.append(loop_scores_df)
                
            except Exception as e:
                print(f"Sample {sample_idx} failed: {e}")
                continue
        
        # Combine all results
        if all_loop_scores:
            combined_df = pd.concat(all_loop_scores, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_dominant_loop_at_time(self, t, loop_scores_df=None):
        """
        Get the dominant loop at a specific time.
        
        Args:
            t: Time point
            loop_scores_df: DataFrame of loop scores (uses self.loop_scores_df if None)
        
        Returns:
            Tuple of (loop_name, relative_score)
        """
        if loop_scores_df is None:
            loop_scores_df = self.loop_scores_df
        
        if loop_scores_df is None:
            raise ValueError("No loop scores available.")
        
        # Find closest time
        times = loop_scores_df['time'].values
        idx = np.argmin(np.abs(times - t))
        row = loop_scores_df.iloc[idx]
        
        # Find loop with highest absolute relative score
        max_score = 0
        dominant_loop = None
        
        for loop_name in self.loop_names:
            col = f'{loop_name}_relative'
            if col in row:
                score = abs(row[col])
                if score > max_score:
                    max_score = score
                    dominant_loop = loop_name
        
        return dominant_loop, row.get(f'{dominant_loop}_relative', 0)
    
    def get_loop_summary(self, loop_scores_df=None):
        """
        Get a summary of loop dominance over the simulation.
        
        Returns:
            DataFrame with summary statistics for each loop
        """
        if loop_scores_df is None:
            loop_scores_df = self.loop_scores_df
        
        summary_data = []
        
        for loop_idx, loop_name in enumerate(self.loop_names):
            rel_col = f'{loop_name}_relative'
            raw_col = f'{loop_name}_raw'
            
            if rel_col in loop_scores_df.columns:
                rel_scores = loop_scores_df[rel_col].values
                raw_scores = loop_scores_df[raw_col].values if raw_col in loop_scores_df.columns else rel_scores
                
                # Handle NaN values
                rel_scores_clean = rel_scores[~np.isnan(rel_scores)]
                raw_scores_clean = raw_scores[~np.isnan(raw_scores)]
                
                summary_data.append({
                    'loop_name': loop_name,
                    'loop_type': 'Reinforcing' if loop_name.startswith('R') else 'Balancing',
                    'loop_variables': ' → '.join(self.feedback_loops[loop_idx]),
                    'mean_relative': np.mean(rel_scores_clean) if len(rel_scores_clean) > 0 else np.nan,
                    'max_abs_relative': np.max(np.abs(rel_scores_clean)) if len(rel_scores_clean) > 0 else np.nan,
                    'mean_raw': np.mean(raw_scores_clean) if len(raw_scores_clean) > 0 else np.nan,
                    'max_abs_raw': np.max(np.abs(raw_scores_clean)) if len(raw_scores_clean) > 0 else np.nan,
                    'std_relative': np.std(rel_scores_clean) if len(rel_scores_clean) > 0 else np.nan,
                    'n_valid': len(rel_scores_clean),
                    'n_total': len(rel_scores)
                })
        
        return pd.DataFrame(summary_data)
    
    def get_link_score_summary(self, ltm_link_scores_df=None):
        """
        Get a summary of LTM link scores to verify correct implementation.
        
        This is useful for checking that link scores can exceed 1 in magnitude,
        which is a key property of the LTM method.
        
        Returns:
            DataFrame with summary statistics for each link
        """
        if ltm_link_scores_df is None:
            ltm_link_scores_df = self.ltm_link_scores_df
        
        if ltm_link_scores_df is None:
            raise ValueError("No link scores available. Call compute_link_scores first.")
        
        summary_data = []
        links = ltm_link_scores_df.groupby(['source', 'target'])
        
        for (source, target), group in links:
            ltm_scores = group['ltm_link_score'].values
            rel_scores = group['relative_link_score'].values
            is_stock = group['is_stock_target'].iloc[0]
            
            summary_data.append({
                'source': source,
                'target': target,
                'is_stock_target': is_stock,
                'ltm_mean': np.mean(ltm_scores),
                'ltm_max_abs': np.max(np.abs(ltm_scores)),
                'ltm_min': np.min(ltm_scores),
                'ltm_max': np.max(ltm_scores),
                'rel_mean': np.mean(rel_scores),
                'rel_max_abs': np.max(np.abs(rel_scores)),
                'exceeds_1': np.any(np.abs(ltm_scores) > 1.0)
            })
        
        return pd.DataFrame(summary_data)
    
    def print_loops(self):
        """Print all identified feedback loops."""
        print(f"\nIdentified {len(self.feedback_loops)} feedback loops:\n")
        
        for loop_idx, (loop, name) in enumerate(zip(self.feedback_loops, self.loop_names)):
            loop_type = "Reinforcing" if name.startswith('R') else "Balancing"
            loop_str = ' → '.join(loop) + ' → ' + loop[0]
            print(f"  {name} ({loop_type}): {loop_str}")

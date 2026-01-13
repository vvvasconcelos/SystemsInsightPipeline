"""
Loops That Matter (LTM) - Pathway-based loop dominance analysis for System Dynamics models.

Based on the theory described by Schoenberg, Davidsen and Eberlein (2019):
https://arxiv.org/abs/1908.11434

This module implements quantitative loop dominance analysis that:
1. Computes link scores: contribution of each causal link to variable changes at each time step
2. Combines link scores into loop scores: product of link scores around feedback loops
3. Normalizes into relative loop scores: bounded measure (-1 to 1) of loop importance

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
    
    This class provides pathway-based loop dominance analysis, identifying which
    feedback loops are responsible for generating observed model behavior at each
    point in time.
    
    Attributes:
        sdm: Reference to the SDM model instance
        feedback_loops: List of feedback loops in the model (each loop is a list of variable names)
        loop_names: List of loop names (e.g., "R1", "B1", etc.)
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
        
        # Store results
        self.link_scores_df = None
        self.loop_scores_df = None
        
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
    
    def compute_link_scores(self, df_sol, params, t_eval):
        """
        Compute link scores for all links at each time step.
        
        The link score measures how much of the change in a target variable
        is attributable to a source variable.
        
        Link score = (∂target/∂source) * Δsource / Σ|all contributions to target|
        
        Args:
            df_sol: Solution dataframe with variable values at each time step
            params: Parameter dictionary
            t_eval: Array of time points
        
        Returns:
            DataFrame with link scores indexed by time, columns are (source, target) tuples
        """
        link_scores = {}
        
        # Get all active links from the adjacency matrix
        active_links = []
        for target in self.df_adj.index:
            for source in self.df_adj.columns:
                if self.df_adj.loc[target, source] != 0:
                    active_links.append((source, target))
        
        # Initialize storage
        for link in active_links:
            link_scores[link] = []
        
        # Compute link scores for each time step (need at least 2 points)
        n_times = len(t_eval)
        
        for t_idx in range(1, n_times):
            t = t_eval[t_idx]
            t_prev = t_eval[t_idx - 1]
            dt = t - t_prev
            
            # Get variable values at current and previous time
            for link in active_links:
                source, target = link
                
                # Get coefficient (partial derivative)
                coef = self._get_link_coefficient(source, target, params)
                
                # Get source change
                if source in df_sol.columns:
                    source_curr = df_sol.loc[t, source]
                    source_prev = df_sol.loc[t_prev, source]
                    delta_source = source_curr - source_prev
                else:
                    delta_source = 0.0
                
                # Get target change
                if target in df_sol.columns:
                    target_curr = df_sol.loc[t, target]
                    target_prev = df_sol.loc[t_prev, target]
                    delta_target = target_curr - target_prev
                else:
                    delta_target = 0.0
                
                # Compute contribution of this link to target change
                contribution = coef * delta_source
                
                # Store raw contribution (we'll normalize later per target)
                link_scores[link].append({
                    'time': t,
                    'contribution': contribution,
                    'delta_target': delta_target
                })
        
        # Convert to DataFrame and normalize per target
        link_scores_data = []
        
        for t_idx in range(len(t_eval) - 1):
            t = t_eval[t_idx + 1]
            
            # Group by target to normalize
            target_contributions = {}
            
            for link in active_links:
                source, target = link
                contrib_data = link_scores[link][t_idx]
                
                if target not in target_contributions:
                    target_contributions[target] = []
                target_contributions[target].append({
                    'link': link,
                    'contribution': contrib_data['contribution'],
                    'delta_target': contrib_data['delta_target']
                })
            
            # Normalize and store
            for target, contribs in target_contributions.items():
                total_abs_contrib = sum(abs(c['contribution']) for c in contribs)
                
                for c in contribs:
                    link = c['link']
                    source = link[0]
                    
                    if total_abs_contrib > 1e-10:
                        relative_score = c['contribution'] / total_abs_contrib
                    else:
                        relative_score = 0.0
                    
                    link_scores_data.append({
                        'time': t,
                        'source': source,
                        'target': target,
                        'link_score': relative_score,
                        'raw_contribution': c['contribution'],
                        'delta_target': c['delta_target']
                    })
        
        self.link_scores_df = pd.DataFrame(link_scores_data)
        return self.link_scores_df
    
    def compute_loop_scores(self, link_scores_df=None):
        """
        Compute loop scores by multiplying link scores around each feedback loop.
        
        The loop score is the product of link scores for all links in the loop.
        Relative loop scores are normalized to sum to 1 (in absolute value).
        
        Args:
            link_scores_df: DataFrame of link scores (uses self.link_scores_df if None)
        
        Returns:
            DataFrame with loop scores indexed by time, one column per loop
        """
        if link_scores_df is None:
            link_scores_df = self.link_scores_df
        
        if link_scores_df is None:
            raise ValueError("No link scores available. Call compute_link_scores first.")
        
        times = link_scores_df['time'].unique()
        
        loop_scores_data = []
        
        for t in times:
            t_data = link_scores_df[link_scores_df['time'] == t]
            
            # Create lookup for link scores at this time
            link_lookup = {}
            for _, row in t_data.iterrows():
                link_lookup[(row['source'], row['target'])] = row['link_score']
            
            # Compute score for each loop
            loop_raw_scores = {}
            
            for loop_idx, loop in enumerate(self.feedback_loops):
                loop_name = self.loop_names[loop_idx]
                n = len(loop)
                
                # Product of link scores around the loop
                loop_score = 1.0
                all_links_present = True
                
                for i in range(n):
                    source = loop[i]
                    target = loop[(i + 1) % n]
                    link = (source, target)
                    
                    if link in link_lookup:
                        loop_score *= link_lookup[link]
                    else:
                        # Link not in scores, might be through auxiliaries
                        # Use adjacency matrix sign
                        adj_val = self.df_adj.loc[target, source]
                        if adj_val != 0:
                            loop_score *= np.sign(adj_val) if adj_val != -999 else 1.0
                        else:
                            all_links_present = False
                            break
                
                if all_links_present:
                    loop_raw_scores[loop_name] = loop_score
                else:
                    loop_raw_scores[loop_name] = 0.0
            
            # Normalize loop scores
            total_abs = sum(abs(s) for s in loop_raw_scores.values())
            
            row_data = {'time': t}
            for loop_name, score in loop_raw_scores.items():
                if total_abs > 1e-10:
                    row_data[f'{loop_name}_relative'] = score / total_abs
                else:
                    row_data[f'{loop_name}_relative'] = 0.0
                row_data[f'{loop_name}_raw'] = score
            
            loop_scores_data.append(row_data)
        
        self.loop_scores_df = pd.DataFrame(loop_scores_data)
        return self.loop_scores_df
    
    def run_ltm_analysis(self, params, t_start=0, t_end=None, n_points=100, x0=None, 
                         constants_values=None, progress_bar=True):
        """
        Run a complete LTM analysis: simulate the model and compute all scores.
        
        Args:
            params: Parameter dictionary for the model
            t_start: Start time for simulation
            t_end: End time for simulation (uses model default if None)
            n_points: Number of time points to evaluate
            x0: Initial conditions for stocks (zeros if None)
            constants_values: Values for constants (zeros if None)
            progress_bar: Whether to show progress bar
        
        Returns:
            Tuple of (solution_df, link_scores_df, loop_scores_df)
        """
        if t_end is None:
            t_end = self.sdm.t_span[1]
        
        t_eval = np.linspace(t_start, t_end, n_points)
        
        # Set up initial conditions
        if x0 is None:
            x0 = np.zeros(len(self.stocks), dtype=np.float64)
        if constants_values is None:
            constants_values = np.zeros(len(self.constants), dtype=np.float64)
        
        # Get system matrices
        if self.sdm.interaction_terms:
            A, b = None, None
        else:
            A, b = self.sdm.params_to_A_b(params, constants_values)
        
        # Store original t_eval and set new one for detailed simulation
        original_t_eval = self.sdm.t_eval
        self.sdm.t_eval = t_eval
        
        # Run simulation
        df_sol = self.sdm.run_SDM(x0, constants_values, A, b, params)
        
        # Restore original t_eval
        self.sdm.t_eval = original_t_eval
        
        # Compute link and loop scores
        link_scores_df = self.compute_link_scores(df_sol, params, t_eval)
        loop_scores_df = self.compute_loop_scores()
        
        return df_sol, link_scores_df, loop_scores_df
    
    def run_ltm_across_samples(self, n_samples=None, t_start=0, t_end=None, n_points=50):
        """
        Run LTM analysis across multiple parameter samples and aggregate results.
        
        This pools the loop scores across different parameter realizations to
        understand robust patterns of loop dominance.
        
        Args:
            n_samples: Number of parameter samples (uses model N if None)
            t_start: Start time for simulation
            t_end: End time for simulation
            n_points: Number of time points per simulation
        
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
                # Run LTM analysis
                _, _, loop_scores_df = self.run_ltm_analysis(
                    params, t_start, t_end, n_points, progress_bar=False
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
            col = f'{loop_name}_relative'
            if col in loop_scores_df.columns:
                scores = loop_scores_df[col].values
                
                summary_data.append({
                    'loop_name': loop_name,
                    'loop_type': 'Reinforcing' if loop_name.startswith('R') else 'Balancing',
                    'loop_variables': ' → '.join(self.feedback_loops[loop_idx]),
                    'mean_score': np.mean(scores),
                    'max_abs_score': np.max(np.abs(scores)),
                    'std_score': np.std(scores),
                    'time_dominant': np.sum(np.abs(scores) == np.max(np.abs(loop_scores_df[[f'{ln}_relative' for ln in self.loop_names if f'{ln}_relative' in loop_scores_df.columns]].values), axis=1)) / len(scores)
                })
        
        return pd.DataFrame(summary_data)
    
    def print_loops(self):
        """Print all identified feedback loops."""
        print(f"\nIdentified {len(self.feedback_loops)} feedback loops:\n")
        
        for loop_idx, (loop, name) in enumerate(zip(self.feedback_loops, self.loop_names)):
            loop_type = "Reinforcing" if name.startswith('R') else "Balancing"
            loop_str = ' → '.join(loop) + ' → ' + loop[0]
            print(f"  {name} ({loop_type}): {loop_str}")

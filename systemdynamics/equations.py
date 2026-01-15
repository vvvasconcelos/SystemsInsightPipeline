"""
Custom equation parsing and evaluation module for System Dynamics Models.

This module handles:
1. Parsing custom equations from the Equation column in Kumu Excel files
2. Validating equations against incoming links (warning for mismatches)
3. Evaluating equations with parameter sampling for # symbols
4. Supporting standard numpy functions

Equation Syntax:
- Variable names: Use variable names as they appear in the model
- Parameters: Use # symbol for parameters to be sampled (always positive)
- Functions: Use np.FUNCTION syntax (e.g., np.exp, np.sin, np.log, np.tanh)
- Operators: Standard Python operators (+, -, *, /, **, etc.)

Example equations:
- "#1 * A + #2 * B"  -> Two parameters multiplying variables A and B
- "np.exp(-#1 * A) * B"  -> Exponential decay with parameter #1
- "#1 * np.tanh(#2 * (A - B))"  -> Bounded S-curve function
"""

import re
import warnings
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any


class EquationParser:
    """
    Parser for custom equations in System Dynamics Models.
    
    Handles parsing, validation, and preparation of equations for evaluation.
    """
    
    # Allowed numpy functions (can be extended)
    ALLOWED_NP_FUNCTIONS = {
        'exp', 'log', 'log10', 'log2',
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
        'sqrt', 'abs', 'sign',
        'floor', 'ceil', 'round',
        'maximum', 'minimum', 'clip',
        'power', 'square',
        'sigmoid',  # Custom sigmoid function we'll add
    }
    
    # Pattern to match parameter placeholders (#, #1, #2, etc.)
    PARAM_PATTERN = re.compile(r'#(\d*)')
    
    # Pattern to match numpy function calls
    NP_FUNC_PATTERN = re.compile(r'np\.(\w+)')
    
    def __init__(self, variables: List[str]):
        """
        Initialize the equation parser.
        
        Args:
            variables: List of all variable names in the model
        """
        self.variables = set(variables)
        self.variable_list = variables
        
    def parse_equation(self, var_name: str, equation: str, incoming_links: List[str]) -> Dict:
        """
        Parse and validate a custom equation.
        
        Args:
            var_name: Name of the variable this equation defines
            equation: The equation string from the Equation column
            incoming_links: List of variables that have incoming links to this variable
            
        Returns:
            dict with keys:
                - 'equation': Original equation string
                - 'variables_used': Set of variable names found in equation
                - 'n_parameters': Number of parameters (# symbols)
                - 'parameter_indices': List of parameter indices found
                - 'validation_warnings': List of warning messages
                - 'is_valid': Boolean indicating if equation is valid for evaluation
                - 'compiled_equation': Prepared equation string for eval()
        """
        result = {
            'equation': equation,
            'variables_used': set(),
            'n_parameters': 0,
            'parameter_indices': [],
            'validation_warnings': [],
            'is_valid': True,
            'compiled_equation': None
        }
        
        if not equation or str(equation).lower() == 'nan' or equation.strip() == '':
            result['is_valid'] = False
            return result
        
        equation = str(equation).strip()
        
        # 1. Find all parameter placeholders
        param_matches = list(self.PARAM_PATTERN.finditer(equation))
        param_indices = []
        for match in param_matches:
            idx = match.group(1)
            if idx == '':
                # Bare # - assign sequential index
                param_indices.append(None)  # Will be assigned later
            else:
                param_indices.append(int(idx))
        
        # Assign sequential indices to bare # symbols
        next_idx = max([i for i in param_indices if i is not None], default=0) + 1
        for i, idx in enumerate(param_indices):
            if idx is None:
                param_indices[i] = next_idx
                next_idx += 1
        
        result['parameter_indices'] = sorted(set(param_indices))
        result['n_parameters'] = len(result['parameter_indices'])
        
        # 2. Find all variables used in the equation
        # Remove numpy function calls and operators to isolate variable names
        temp_eq = equation
        temp_eq = self.NP_FUNC_PATTERN.sub('', temp_eq)  # Remove np.func
        temp_eq = self.PARAM_PATTERN.sub('', temp_eq)    # Remove #params
        
        # Find words that could be variable names
        word_pattern = re.compile(r'\b([A-Za-z][A-Za-z0-9_\s]*)\b')
        potential_vars = word_pattern.findall(temp_eq)
        
        for pv in potential_vars:
            pv_clean = pv.strip()
            if pv_clean in self.variables:
                result['variables_used'].add(pv_clean)
            elif pv_clean and pv_clean not in ['np', 'True', 'False', 'None']:
                # Check if it might be a partial match due to word boundaries
                for var in self.variables:
                    if var in temp_eq:
                        result['variables_used'].add(var)
        
        # Also do direct matching for multi-word variables
        for var in self.variables:
            if var in equation:
                result['variables_used'].add(var)
        
        # 3. Validate numpy functions used
        np_funcs = self.NP_FUNC_PATTERN.findall(equation)
        for func in np_funcs:
            if func not in self.ALLOWED_NP_FUNCTIONS:
                result['validation_warnings'].append(
                    f"Unknown numpy function 'np.{func}'. Allowed functions: {sorted(self.ALLOWED_NP_FUNCTIONS)}"
                )
        
        # 4. Validate against incoming links
        incoming_set = set(incoming_links)
        vars_used = result['variables_used']
        
        # Variables in equation but not in incoming links
        extra_vars = vars_used - incoming_set
        if extra_vars:
            result['validation_warnings'].append(
                f"Variable(s) {extra_vars} used in equation but not defined as incoming links to '{var_name}'"
            )
        
        # Incoming links not used in equation
        missing_vars = incoming_set - vars_used
        if missing_vars:
            result['validation_warnings'].append(
                f"Incoming link(s) {missing_vars} to '{var_name}' not used in equation"
            )
        
        # 5. Prepare compiled equation
        result['compiled_equation'] = self._prepare_equation_for_eval(equation, result['parameter_indices'])
        
        return result
    
    def _prepare_equation_for_eval(self, equation: str, param_indices: List[int]) -> str:
        """
        Prepare equation string for evaluation with eval().
        
        Converts:
        - Variable names -> vals['Variable Name']
        - #N -> params[N-1]
        - Bare # -> params[sequential_index]
        
        Args:
            equation: Original equation string
            param_indices: List of parameter indices used
            
        Returns:
            String ready for eval() with vals and params dicts
        """
        compiled = equation
        
        # Sort variables by length (longest first) to avoid partial replacements
        sorted_vars = sorted(self.variables, key=len, reverse=True)
        
        # Replace variable names with vals['varname']
        for var in sorted_vars:
            if var in compiled:
                # Use word boundary matching to avoid partial replacements
                # But handle multi-word variables carefully
                pattern = re.escape(var)
                compiled = re.sub(
                    rf'(?<!["\'\[])\b{pattern}\b(?!["\'\]])',
                    f"vals['{var}']",
                    compiled
                )
        
        # Replace parameters: #N -> params[N-1] (0-indexed in params list)
        # First replace numbered parameters
        for idx in sorted(param_indices, reverse=True):  # Reverse to handle #10 before #1
            compiled = re.sub(rf'#\b{idx}\b', f'params[{idx-1}]', compiled)
        
        # Replace bare # with sequential params
        param_counter = [0]  # Use list to allow modification in closure
        def replace_bare_param(match):
            if match.group(1) == '':
                result = f'params[{param_counter[0]}]'
                param_counter[0] += 1
                return result
            return match.group(0)
        
        # Only replace bare # that weren't already handled
        compiled = re.sub(r'#(?!\d)(?!\[)', lambda m: f'params[{param_counter[0]}]', compiled, count=0)
        
        return compiled


class EquationEvaluator:
    """
    Evaluates custom equations during simulation.
    """
    
    def __init__(self, equations: Dict[str, Dict], parameter_range: Tuple[float, float] = (0, 1)):
        """
        Initialize the equation evaluator.
        
        Args:
            equations: Dictionary mapping variable names to parsed equation dicts
            parameter_range: (min, max) range for sampling parameters
        """
        self.equations = equations
        self.parameter_range = parameter_range
        
        # Custom functions to add to evaluation namespace
        self.custom_funcs = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        }
    
    def sample_equation_parameters(self, var_name: str) -> np.ndarray:
        """
        Sample parameters for a variable's equation.
        
        Args:
            var_name: Name of the variable
            
        Returns:
            Array of sampled parameter values (all positive)
        """
        if var_name not in self.equations:
            return np.array([])
        
        eq_info = self.equations[var_name]
        n_params = eq_info['n_parameters']
        
        if n_params == 0:
            return np.array([])
        
        # Sample from uniform distribution in positive range
        return np.random.uniform(
            self.parameter_range[0], 
            self.parameter_range[1], 
            size=n_params
        )
    
    def evaluate(self, var_name: str, vals: Dict[str, float], params: np.ndarray) -> float:
        """
        Evaluate a custom equation.
        
        Args:
            var_name: Name of the variable
            vals: Dictionary of current variable values
            params: Array of parameter values for this equation
            
        Returns:
            Computed value
        """
        if var_name not in self.equations:
            raise ValueError(f"No equation defined for variable '{var_name}'")
        
        eq_info = self.equations[var_name]
        compiled = eq_info['compiled_equation']
        
        if not compiled:
            raise ValueError(f"Equation for '{var_name}' could not be compiled")
        
        # Build evaluation namespace
        eval_namespace = {
            'np': np,
            'vals': vals,
            'params': params,
            **self.custom_funcs
        }
        
        try:
            result = eval(compiled, {"__builtins__": {}}, eval_namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating equation for '{var_name}': {e}\nEquation: {eq_info['equation']}\nCompiled: {compiled}")
    
    def has_custom_equation(self, var_name: str) -> bool:
        """Check if a variable has a custom equation defined."""
        return var_name in self.equations and self.equations[var_name]['is_valid']


def parse_equations_from_dataframe(df_elements, variables: List[str], 
                                   adjacency_matrix, variable_list: List[str]) -> Tuple[Dict, List[str]]:
    """
    Parse equations from an Elements dataframe.
    
    Args:
        df_elements: DataFrame with columns including 'Label' and 'Equation'
        variables: List of cleaned variable names
        adjacency_matrix: DataFrame with adjacency matrix
        variable_list: Ordered list of all variables
        
    Returns:
        Tuple of (equations_dict, all_warnings)
    """
    parser = EquationParser(variables)
    equations = {}
    all_warnings = []
    
    if 'Equation' not in df_elements.columns:
        return equations, all_warnings
    
    for idx, row in df_elements.iterrows():
        label = row['Label']
        equation = row.get('Equation', None)
        
        if pd.isna(equation) or str(equation).strip() == '':
            continue
        
        # Find cleaned variable name
        var_name = None
        for v in variables:
            if v == label or label.strip() == v:
                var_name = v
                break
        
        if var_name is None:
            continue
        
        # Get incoming links from adjacency matrix
        if var_name in adjacency_matrix.index:
            incoming_links = [
                col for col in adjacency_matrix.columns 
                if adjacency_matrix.loc[var_name, col] != 0
            ]
        else:
            incoming_links = []
        
        # Parse the equation
        parsed = parser.parse_equation(var_name, equation, incoming_links)
        
        if parsed['is_valid']:
            equations[var_name] = parsed
            
            # Collect warnings
            for warning in parsed['validation_warnings']:
                warning_msg = f"[{var_name}] {warning}"
                all_warnings.append(warning_msg)
                warnings.warn(warning_msg)
    
    return equations, all_warnings


# For pandas import in parse_equations_from_dataframe
import pandas as pd

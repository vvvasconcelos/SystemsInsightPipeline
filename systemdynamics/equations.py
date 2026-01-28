"""
Custom equation parsing and evaluation module for System Dynamics Models.

This module handles:
1. Parsing custom equations from the Equation column in Kumu Excel files
2. Validating equations against incoming links (warning for mismatches)
3. Evaluating equations with parameter sampling for # symbols
4. Supporting standard numpy functions
5. Handling intervention parameters with $ symbol

Equation Syntax:
- Variable names: Use variable names as they appear in the model
- Parameters: Use # symbol for parameters to be sampled (always positive)
- Intervention: Use $ symbol to indicate where the intervention is applied
- Functions: Use np.FUNCTION syntax (e.g., np.exp, np.sin, np.log, np.tanh)
- Operators: Standard Python operators (+, -, *, /, **, etc.)

Example equations:
- "#1 * A + #2 * B"  -> Two parameters multiplying variables A and B
- "np.exp(-#1 * A) * B"  -> Exponential decay with parameter #1
- "#1 * np.tanh(#2 * (A - B))"  -> Bounded S-curve function
- "$ + #1 * np.tanh(#2 * A)"  -> Intervention added before S-curve
- "#1 * ($ + A)"  -> Intervention combined with variable A

If no $ is specified in the equation, the intervention is added to the result
(i.e., result = equation_value + intervention_intensity).
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
    
    # Pattern to match intervention placeholder ($)
    INTERVENTION_PATTERN = re.compile(r'\$')
    
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
                - 'has_intervention': Boolean indicating if $ is in equation
                - 'validation_warnings': List of warning messages
                - 'is_valid': Boolean indicating if equation is valid for evaluation
                - 'compiled_equation': Prepared equation string for eval()
        """
        result = {
            'equation': equation,
            'variables_used': set(),
            'n_parameters': 0,
            'parameter_indices': [],
            'param_index_map': {},  # Maps original #N to consecutive index
            'has_intervention': False,
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
        
        # Create mapping from original parameter indices to consecutive indices
        # e.g., if equation uses #1, #3, #7, map to params[0], params[1], params[2]
        result['param_index_map'] = {orig_idx: new_idx for new_idx, orig_idx in enumerate(result['parameter_indices'])}
        
        # 2. Check for intervention placeholder ($)
        intervention_matches = self.INTERVENTION_PATTERN.findall(equation)
        result['has_intervention'] = len(intervention_matches) > 0
        if len(intervention_matches) > 1:
            result['validation_warnings'].append(
                f"Multiple $ symbols found in equation. Only one intervention placeholder is supported."
            )
        
        # 3. Find all variables used in the equation
        # Remove numpy function calls and operators to isolate variable names
        temp_eq = equation
        temp_eq = self.NP_FUNC_PATTERN.sub('', temp_eq)  # Remove np.func
        temp_eq = self.PARAM_PATTERN.sub('', temp_eq)    # Remove #params
        temp_eq = self.INTERVENTION_PATTERN.sub('', temp_eq)  # Remove $ intervention
        
        # Find words that could be variable names
        word_pattern = re.compile(r'\b([A-Za-z][A-Za-z0-9_\s]*)\b')
        potential_vars = word_pattern.findall(temp_eq)
        
        # Track unrecognized identifiers for error checking
        unrecognized_identifiers = set()
        
        # Pattern to detect likely parameter typos (p1, p2, etc. instead of #1, #2)
        param_typo_pattern = re.compile(r'^p\d+$')
        
        for pv in potential_vars:
            pv_clean = pv.strip()
            if pv_clean in self.variables:
                result['variables_used'].add(pv_clean)
            elif pv_clean and pv_clean not in ['np', 'True', 'False', 'None', '']:
                # Check if it looks like a parameter typo (p1, p2, etc.)
                if param_typo_pattern.match(pv_clean):
                    result['validation_warnings'].append(
                        f"Found '{pv_clean}' which looks like a parameter. "
                        f"Did you mean '#{pv_clean[1:]}' (with # instead of p)?"
                    )
                    unrecognized_identifiers.add(pv_clean)
                else:
                    # Check if it might be a partial match due to word boundaries
                    found_match = False
                    for var in self.variables:
                        if var in temp_eq:
                            result['variables_used'].add(var)
                            found_match = True
                    if not found_match and len(pv_clean) > 1:  # Ignore single letters that might be math variables
                        unrecognized_identifiers.add(pv_clean)
        
        # Also do direct matching for multi-word variables
        for var in self.variables:
            if var in equation:
                result['variables_used'].add(var)
        
        # 3. Check for unrecognized variable names (ERROR - will fail at runtime)
        if unrecognized_identifiers:
            # Filter out common false positives
            filtered_unrecognized = {u for u in unrecognized_identifiers 
                                     if u.lower() not in {'x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'n', 'm'}}
            if filtered_unrecognized:
                result['is_valid'] = False
                result['validation_warnings'].append(
                    f"ERROR: Unrecognized variable(s) {filtered_unrecognized} in equation. "
                    f"These do not exist in the model and will cause a runtime error."
                )
        
        # 4. Validate numpy functions used
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
        - $ -> intervention (intervention intensity variable)
        - Bare function names (tanh, exp, etc.) -> np.function
        
        Args:
            equation: Original equation string
            param_indices: List of parameter indices used
            
        Returns:
            String ready for eval() with vals and params dicts
        """
        compiled = equation
        
        # First, convert bare function names to np.function (e.g., tanh -> np.tanh)
        # Only convert if not already prefixed with np.
        for func in self.ALLOWED_NP_FUNCTIONS:
            # Match function name followed by "(" but not preceded by "np."
            # Use simple word boundary and check for np. prefix
            compiled = re.sub(rf'(?<!np\.)\b{func}\s*\(', f'np.{func}(', compiled)
        
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
        
        # Replace intervention placeholder: $ -> intervention
        compiled = re.sub(r'\$', 'intervention', compiled)
        
        # Create mapping from original parameter indices to consecutive indices
        # e.g., if equation uses #1, #3, #7, map to params[0], params[1], params[2]
        param_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted(param_indices))}
        
        # Replace parameters: #N -> params['#N'] (reverse order to handle #10 before #1)
        for idx in sorted(param_indices, reverse=True):
            compiled = re.sub(rf'#\b{idx}\b', f"params['#{idx}']", compiled)

        # Replace bare # with sequential params (params['#1'], params['#2'], ...)
        param_counter = [1]  # Start at 1 for #1
        def replace_bare_param(match):
            if match.group(1) == '':
                result = f"params['#{param_counter[0]}']"
                param_counter[0] += 1
                return result
            return match.group(0)
        compiled = re.sub(r'#(?!\d)(?!\[)', replace_bare_param, compiled)
        
        return compiled


class EquationEvaluator:
    """
    Evaluates custom equations during simulation.
    """

    def get_equation_info(self, var_name):
        """
        Return the parsed equation info dict for the given variable name.
        """
        return self.equations.get(var_name, {})
    
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
    
#    def sample_equation_parameters(self, var_name: str) -> np.ndarray:
#        """
#        Sample parameters for a variable's equation.
#        
#        Args:
#            var_name: Name of the variable
#            
#        Returns:
#            Array of sampled parameter values (all positive)
#        """
#        if var_name not in self.equations:
#            return np.array([])
#        
#        eq_info = self.equations[var_name]
#        n_params = eq_info['n_parameters']
#        
#        if n_params == 0:
#            return np.array([])
#        
#        # Sample from uniform distribution in positive range
#        return np.random.uniform(
#            self.parameter_range[0], 
#            self.parameter_range[1], 
#            size=n_params
#        )

    def sample_equation_parameters(self, var_name: str) -> dict:
        if var_name not in self.equations:
            return {}
        eq_info = self.equations[var_name]
        param_indices = eq_info['parameter_indices']
        param_labels = [f'#{idx}' for idx in param_indices]
        n_params = len(param_labels)
        if n_params == 0:
            return {}
        sampled = np.random.uniform(self.parameter_range[0], self.parameter_range[1], size=n_params)
        return {label: val for label, val in zip(param_labels, sampled)}
    
    def evaluate(self, var_name: str, vals: Dict[str, float], params: np.ndarray, 
                 intervention: float = 0.0) -> float:
        """
        Evaluate a custom equation.
        
        Args:
            var_name: Name of the variable
            vals: Dictionary of current variable values
            params: Array of parameter values for this equation
            intervention: Intervention intensity for this variable (default 0.0)
            
        Returns:
            Computed value
            
        Note:
            If the equation contains a $ symbol, the intervention is used where specified.
            If no $ is in the equation, the intervention is added to the final result.
        """
        if var_name not in self.equations:
            raise ValueError(f"No equation defined for variable '{var_name}'")
        
        eq_info = self.equations[var_name]
        compiled = eq_info['compiled_equation']
        has_intervention = eq_info.get('has_intervention', False)
        
        if not compiled:
            raise ValueError(f"Equation for '{var_name}' could not be compiled")
        
        # Build evaluation namespace
        eval_namespace = {
            'np': np,
            'vals': vals,
            'params': params,
            'intervention': intervention,
            **self.custom_funcs
        }
        
        try:
            result = eval(compiled, {"__builtins__": {}}, eval_namespace)
            result = float(result)
            
            # If no $ in equation, add intervention to the result as default behavior
            if not has_intervention and intervention != 0.0:
                result = result + intervention
            
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating equation for '{var_name}': {e}\nEquation: {eq_info['equation']}\nCompiled: {compiled}")
    
    def has_custom_equation(self, var_name: str) -> bool:
        """Check if a variable has a custom equation defined."""
        return var_name in self.equations and self.equations[var_name]['is_valid']
    
    def has_intervention_placeholder(self, var_name: str) -> bool:
        """Check if a variable's equation has a $ intervention placeholder."""
        if var_name not in self.equations:
            return False
        return self.equations[var_name].get('has_intervention', False)
    
    def get_intervention_variables(self) -> List[str]:
        """Get list of variables that have equations with intervention placeholders ($)."""
        return [var for var in self.equations 
                if self.equations[var].get('has_intervention', False) 
                and self.equations[var]['is_valid']]


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

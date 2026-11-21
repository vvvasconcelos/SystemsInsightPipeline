"""
Custom equations for System Dynamics Models.

An ``Equation`` cell in the Elements sheet overrides the default linear combination of a
variable's incoming links. This module parses, validates and compiles those equations, and
evaluates them during simulation.

Equation syntax
---------------
- **Variables**: model variable names exactly as they appear in the Elements sheet. Names may
  contain spaces and punctuation such as ``.``, ``(``, ``)``, ``'`` or ``-`` (only ``+`` and
  ``*`` are not allowed in names). Names are recognised as *whole* names, longest first, so
  ``Price`` is not found inside ``Price index``. Runs of whitespace count as one space.
- **Local parameters**: ``#1``, ``#2``, ... or a bare ``#`` (numbered after the highest explicit
  number, left to right). Each is sampled uniformly from ``[0, parameter_value_aux]`` and is
  *local* to its equation: ``#1`` in one equation and ``#1`` in another are different parameters.
- **Named parameters**: ``#rate``, ``#k_decay``, ... (``#`` followed by a letter, then letters,
  digits or underscores). A named parameter is *the same parameter in every equation* that uses
  it (one value per parameter draw). Its range comes from the optional ``Parameters`` sheet
  (``Name | Low | High``) and defaults to ``[0, parameter_value_aux]``.
- **Intervention**: ``$`` marks where the intervention intensity enters. Without ``$`` the
  intervention is added to the result (``result = equation + intensity``).
- **Time**: ``t`` is the simulation time (unless a model variable is itself called ``t``).
- **Functions**: ``np.<function>`` (e.g. ``np.exp``, ``np.minimum``) or the bare names in
  ``EquationParser.ALLOWED_NP_FUNCTIONS`` (``exp``, ``log``, ``tanh`` ...), plus
  ``sigmoid(x) = 1 / (1 + exp(-x))``.
- **Operators**: ``+ - * / ** %``, comparisons (``(S < 0.5)`` evaluates to 1 or 0), ``and``,
  ``or``, ``not`` and conditional expressions ``a if condition else b``.

Example equations::

    #1 * A + #2 * B                  two local parameters
    np.exp(-#1 * A) * B              exponential decay
    #1 * np.tanh(#2 * (A - B))       bounded S-curve
    $ + #1 * (1 - np.exp(-#2 * A))   saturation; intervention added explicitly
    -#outflow * S1                   named parameter shared with another equation
    K * (t > 2)                      a step at t = 2

Validation happens when the workbook is read: names that are not model variables (typos),
unknown functions and syntax errors raise a ``ValueError`` that names the offending text.
"""

import ast
import difflib
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

# Characters that can be part of a Python identifier (ASCII; equations are ASCII in practice).
_IDENT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# '#' followed by digits (local numbered parameter), a name (named parameter), or nothing (bare).
_PARAM_TOKEN = re.compile(r"#(?:(?P<num>\d+)|(?P<name>[A-Za-z][A-Za-z0-9_]*))?")
_NAMED_PARAMETER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# Placeholders used internally while an equation is validated: _v0_ (variable), _p1_ (local
# parameter #1), _n0_ (named parameter), _i_ (intervention).
_PLACEHOLDER = re.compile(r"_(?:[vpn]\d+|i)_")
# runs of words (for error messages); the look-behind skips the 'e' of number literals like 1e-5
_WORD_RUN = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)*")

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.Name, ast.Attribute, ast.Constant, ast.keyword, ast.Tuple, ast.List,
    ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context,
)

_NAME_HINTS = {
    "time": "the simulation time is written t",
    "Time": "the simulation time is written t",
    "TIME": "the simulation time is written t",
    "min": "use np.minimum(a, b)",
    "max": "use np.maximum(a, b)",
    "pi": "use np.pi",
    "e": "use np.e (or np.exp(x))",
}


def normalize_name(name) -> str:
    """Collapse runs of whitespace to single spaces (the rule Extract applies to labels)."""
    return " ".join(str(name).split())


class EquationParser:
    """Parse, validate and compile custom equations.

    Args:
        variables: all variable names of the model.
        var_types: optional ``{variable: 'stock' | 'auxiliary' | 'constant'}``. When given, a
            stock that refers to itself (decay, goal seeking) is accepted silently and an
            auxiliary that refers to itself is rejected (an algebraic loop).
    """

    # numpy functions that may be written without the ``np.`` prefix
    ALLOWED_NP_FUNCTIONS = {
        'exp', 'log', 'log10', 'log2', 'log1p', 'expm1',
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
        'sqrt', 'abs', 'sign',
        'floor', 'ceil', 'round',
        'maximum', 'minimum', 'clip',
        'power', 'square',
        'where', 'heaviside',
    }
    # functions provided by SIP itself (not numpy)
    CUSTOM_FUNCTIONS = {'sigmoid'}
    # numpy constants accepted as ``np.<name>``
    NP_CONSTANTS = {'pi', 'e', 'inf', 'nan', 'euler_gamma'}
    # name of the simulation-time variable inside equations
    TIME_NAMES = {'t'}

    # Kept for backward compatibility (no longer used internally).
    PARAM_PATTERN = re.compile(r'#(\d*)')
    INTERVENTION_PATTERN = re.compile(r'\$')
    NP_FUNC_PATTERN = re.compile(r'np\.(\w+)')

    def __init__(self, variables: List[str], var_types: Optional[Dict[str, str]] = None):
        self.variable_list = list(variables)
        self.variables = set(self.variable_list)
        self.var_types = {k: str(v).lower() for k, v in (var_types or {}).items()}
        # normalized name -> model name, and candidates by first character (longest first)
        self._by_normalized = {}
        for v in self.variable_list:
            self._by_normalized.setdefault(normalize_name(v), v)
        self._candidates = {}
        for norm in sorted(self._by_normalized, key=len, reverse=True):
            if norm:
                self._candidates.setdefault(norm[0], []).append(norm)

    # ------------------------------------------------------------------ public API
    def parse_equation(self, var_name: str, equation: str, incoming_links: List[str]) -> Dict:
        """Parse and validate one equation.

        Returns a dict with keys:
            - 'equation': the equation as given
            - 'variables_used': set of model variables referenced
            - 'n_parameters': number of local (numbered) parameters
            - 'parameter_indices': sorted local parameter numbers (``#1`` -> 1)
            - 'param_index_map': ``{number: position}`` for the local parameters
            - 'named_parameters': sorted names of the named parameters (``#rate`` -> 'rate')
            - 'has_intervention': whether ``$`` occurs
            - 'uses_time': whether the time ``t`` occurs
            - 'validation_warnings': messages; entries starting with ``ERROR:`` make it invalid
            - 'is_valid': False if the equation is empty or has errors
            - 'compiled_equation': Python expression evaluated with ``vals``, ``params``,
              ``intervention`` and ``t`` in scope
        """
        result = {
            'equation': equation,
            'variables_used': set(),
            'n_parameters': 0,
            'parameter_indices': [],
            'param_index_map': {},
            'named_parameters': [],
            'has_intervention': False,
            'uses_time': False,
            'validation_warnings': [],
            'is_valid': True,
            'compiled_equation': None,
        }
        if equation is None or (isinstance(equation, float) and np.isnan(equation)):
            result['is_valid'] = False
            return result
        text = normalize_name(equation)
        if text == '' or text.lower() == 'nan':
            result['is_valid'] = False
            return result

        errors: List[str] = []
        notes: List[str] = result['validation_warnings']

        if any(_PLACEHOLDER.fullmatch(w) for w in re.findall(r"[A-Za-z0-9_]+", text)):
            result['is_valid'] = False
            notes.append("ERROR: names of the form _v0_, _p1_, _n0_ or _i_ are reserved by SIP.")
            return result

        # 1. Replace whole variable names (longest first) by placeholders.
        code, used_vars = self._substitute_variables(text)
        result['variables_used'] = set(used_vars)

        # 2. Parameters: '#k' local, '#name' named, bare '#' numbered after the explicit ones.
        code, local_indices, named = self._substitute_parameters(code)
        result['parameter_indices'] = sorted(local_indices)
        result['n_parameters'] = len(result['parameter_indices'])
        result['param_index_map'] = {idx: k for k, idx in enumerate(result['parameter_indices'])}
        result['named_parameters'] = sorted(named)

        # 3. Intervention placeholder.
        n_dollar = code.count('$')
        result['has_intervention'] = n_dollar > 0
        if n_dollar > 1:
            notes.append("Multiple $ symbols found in equation. Only one intervention placeholder is supported.")
        code = code.replace('$', ' _i_ ')

        # 4. Functions: 'sigmoid' is SIP's own; bare numpy names get the 'np.' prefix.
        code = re.sub(r"(?<![\w.])np\s*\.\s*sigmoid\b", "sigmoid", code)
        for func in sorted(self.ALLOWED_NP_FUNCTIONS):
            code = re.sub(rf"(?<![\w.]){func}\s*\(", f"np.{func}(", code)
        # Tidy the padding around placeholders (the code holds no variable names or strings
        # at this point, so this cannot alter a name).
        code = " ".join(code.split())
        code = re.sub(r"\(\s+", "(", code)
        code = re.sub(r"\s+\)", ")", code)

        # 5. Validate the expression structurally.
        uses_time = self._validate(code.strip(), used_vars, errors, notes)
        result['uses_time'] = uses_time

        # 6. Validate against the diagram (warnings only; I-17 for self references).
        self._check_links(var_name, set(used_vars), incoming_links, errors, notes)

        if errors:
            result['is_valid'] = False
            for e in errors:
                notes.append(f"ERROR: {e}")
            return result

        # 7. Final expression for eval(): placeholders -> runtime look-ups.
        compiled = self._finalize(code, used_vars, named)
        try:
            compile(compiled, f"<equation {var_name}>", "eval")
        except SyntaxError as exc:  # pragma: no cover - guarded by _validate
            result['is_valid'] = False
            notes.append(f"ERROR: the equation could not be compiled ({exc.msg}).")
            return result
        result['compiled_equation'] = compiled
        return result

    # ------------------------------------------------------------------ steps
    def _substitute_variables(self, text: str) -> Tuple[str, List[str]]:
        """Scan ``text`` left to right and replace whole variable names by placeholders."""
        out, used, index = [], [], {}
        i, n = 0, len(text)
        while i < n:
            match = None
            ch = text[i]
            if ch in self._candidates:
                for norm in self._candidates[ch]:          # longest first
                    if text.startswith(norm, i) and self._boundaries_ok(text, i, i + len(norm), norm):
                        match = norm
                        break
            if match is None:
                out.append(ch)
                i += 1
                continue
            name = self._by_normalized[match]
            if name not in index:
                index[name] = len(used)
                used.append(name)
            out.append(f" _v{index[name]}_ ")
            i += len(match)
        return "".join(out), used

    @staticmethod
    def _boundaries_ok(text: str, start: int, end: int, name: str) -> bool:
        """A name must not be glued to identifier characters (or be an attribute / parameter)."""
        if name[0] in _IDENT_CHARS and start > 0:
            prev = text[start - 1]
            if prev in _IDENT_CHARS or prev in '.#':
                return False
        if name[-1] in _IDENT_CHARS and end < len(text):
            nxt = text[end]
            if nxt in _IDENT_CHARS:
                return False
            # 'np.exp' and similar: a name followed by '.<letter>' is an attribute access
            if nxt == '.' and end + 1 < len(text) and (text[end + 1].isalpha() or text[end + 1] == '_'):
                return False
        return True

    @staticmethod
    def _substitute_parameters(code: str):
        explicit = [int(m.group('num')) for m in _PARAM_TOKEN.finditer(code) if m.group('num')]
        next_idx = [max(explicit, default=0) + 1]
        local, named = set(), []

        def repl(m):
            if m.group('num'):
                idx = int(m.group('num'))
                local.add(idx)
                return f" _p{idx}_ "
            if m.group('name'):
                name = m.group('name')
                if name not in named:
                    named.append(name)
                return f" _n{named.index(name)}_ "
            idx = next_idx[0]
            next_idx[0] += 1
            local.add(idx)
            return f" _p{idx}_ "

        return _PARAM_TOKEN.sub(repl, code), local, named

    def _validate(self, code: str, used_vars: List[str], errors: List[str], notes: List[str]) -> bool:
        """Check the placeholder expression; returns whether it uses the time ``t``."""
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as exc:
            unknown = self._unknown_phrases(code)
            if unknown:
                errors.append(self._unknown_message(unknown))
            else:
                errors.append(f"the equation is not a valid expression ({exc.msg}). "
                              "Check brackets and operators (multiplication needs '*').")
            return False

        call_funcs = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        attr_bases = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        unknown, uses_time = [], False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                nid = node.id
                if _PLACEHOLDER.fullmatch(nid):
                    if nid.startswith('_v') and id(node) in call_funcs:
                        var = used_vars[int(nid[2:-1])]
                        errors.append(f"model variable '{var}' is followed by '(' as if it were a "
                                      "function. Add an operator (e.g. '*') or rename the variable.")
                    continue
                if nid == 'np':
                    if id(node) not in attr_bases:
                        errors.append("'np' must be followed by a function, e.g. np.exp(x).")
                    continue
                if nid in self.CUSTOM_FUNCTIONS or nid in self.ALLOWED_NP_FUNCTIONS:
                    if id(node) not in call_funcs:
                        errors.append(f"'{nid}' is a function; call it with brackets, e.g. {nid}(x).")
                    continue
                if nid in self.TIME_NAMES:
                    uses_time = True
                    continue
                if nid not in unknown:
                    unknown.append(nid)
            elif isinstance(node, ast.Attribute):
                if not (isinstance(node.value, ast.Name) and node.value.id == 'np'):
                    errors.append("attribute access (a '.' after a name) is only supported for "
                                  "numpy functions such as np.exp.")
                elif node.attr.startswith('_'):
                    errors.append(f"'np.{node.attr}' is not allowed.")
                elif node.attr in self.ALLOWED_NP_FUNCTIONS or node.attr in self.NP_CONSTANTS:
                    pass
                elif hasattr(np, node.attr) and id(node) in call_funcs and not callable(getattr(np, node.attr)):
                    errors.append(f"'np.{node.attr}' is not a function.")
                elif hasattr(np, node.attr):
                    notes.append(f"Unknown numpy function 'np.{node.attr}'. Allowed functions: "
                                 f"{sorted(self.ALLOWED_NP_FUNCTIONS)}")
                else:
                    errors.append(f"numpy has no function or constant 'np.{node.attr}'.")
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (bool, int, float, complex)):
                    errors.append(f"text {node.value!r} is not allowed in equations.")
            elif not isinstance(node, _ALLOWED_NODES):
                errors.append(f"unsupported syntax ({type(node).__name__}) in equation.")
        if unknown:
            errors.append(self._unknown_message(unknown))
        return uses_time

    def _unknown_phrases(self, code: str) -> List[str]:
        known = set(self.ALLOWED_NP_FUNCTIONS) | self.CUSTOM_FUNCTIONS | self.TIME_NAMES | {'np', 'and', 'or', 'not', 'if', 'else', 'True', 'False'}
        phrases = []
        # drop numpy attribute names (np.exp) before looking for leftover words
        stripped = re.sub(r"\bnp\s*\.\s*[A-Za-z_][A-Za-z0-9_]*", " ", code)
        for m in _WORD_RUN.finditer(stripped):
            words = [w for w in m.group(0).split() if not _PLACEHOLDER.fullmatch(w) and w not in known]
            # numbers like 1e-5 are not words; keep only multi-character or single unknown words
            if words:
                phrase = " ".join(words)
                if phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    def _unknown_message(self, unknown: List[str]) -> str:
        parts = []
        for name in unknown:
            hint = None
            if re.fullmatch(r"p\d+", name):
                hint = f"did you mean '#{name[1:]}' (with # instead of p)?"
            elif name in _NAME_HINTS:
                hint = _NAME_HINTS[name]
            else:
                close = difflib.get_close_matches(name, self.variable_list, n=1, cutoff=0.6)
                if close:
                    hint = f"did you mean '{close[0]}'?"
            parts.append(f"'{name}'" + (f" ({hint})" if hint else ""))
        return ("Unrecognized name(s) in equation: " + ", ".join(parts) + ". Names must match "
                "model variables exactly (they are case-sensitive); parameters are written #1 or "
                "#name, and functions as np.exp(...).")

    def _check_links(self, var_name, used, incoming_links, errors, notes):
        incoming = set(incoming_links)
        vtype = self.var_types.get(var_name)
        extra = used - incoming
        if var_name in extra:
            if vtype == 'stock':
                # A stock's equation is its rate of change: referring to its own level (decay,
                # goal seeking) is the normal case and needs no self-link in the diagram.
                extra.discard(var_name)
            elif vtype == 'auxiliary':
                extra.discard(var_name)
                errors.append(f"auxiliary '{var_name}' refers to itself. Auxiliaries are computed "
                              "instantaneously, so this is an algebraic loop; use a stock to "
                              "accumulate or delay.")
        if extra:
            notes.append(f"Variable(s) {extra} used in equation but not defined as incoming links to '{var_name}'")
        missing = incoming - used
        if missing:
            notes.append(f"Incoming link(s) {missing} to '{var_name}' not used in equation")

    @staticmethod
    def _finalize(code: str, used_vars: List[str], named: List[str]) -> str:
        out = re.sub(r"\b_v(\d+)_\b", lambda m: f"vals[{used_vars[int(m.group(1))]!r}]", code)
        out = re.sub(r"\b_p(\d+)_\b", lambda m: f"params['#{m.group(1)}']", out)
        out = re.sub(r"\b_n(\d+)_\b", lambda m: f"params['#{named[int(m.group(1))]}']", out)
        out = re.sub(r"\b_i_\b", "intervention", out)
        return out


class EquationEvaluator:
    """Evaluates parsed custom equations during simulation.

    Equations are compiled once to code objects (not re-parsed on every call). The parameter
    values of an equation are passed as a dict: ``{'#1': value, '#2': value, '#rate': value}``.
    """

    def __init__(self, equations: Dict[str, Dict], parameter_range: Tuple[float, float] = (0, 1)):
        """
        Args:
            equations: ``{variable: parsed equation dict}`` (see ``EquationParser.parse_equation``)
            parameter_range: (min, max) range for sampling local ``#`` parameters
        """
        self.equations = equations
        self.parameter_range = parameter_range
        # Custom functions available in the evaluation namespace
        self.custom_funcs = {'sigmoid': expit}
        self._init_runtime()

    def _init_runtime(self):
        self._globals = {'__builtins__': {}, 'np': np, **self.custom_funcs}
        self._code = {}

    # pickling support: code objects and modules are rebuilt on load
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_globals', None)
        state.pop('_code', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_runtime()

    def get_equation_info(self, var_name):
        """Return the parsed equation info dict for the given variable name."""
        return self.equations.get(var_name, {})

    def _compiled(self, var_name):
        code = self._code.get(var_name)
        if code is None:
            eq_info = self.equations[var_name]
            compiled = eq_info.get('compiled_equation')
            if not compiled:
                raise ValueError(f"Equation for '{var_name}' could not be compiled")
            code = compile(compiled, f"<equation {var_name}>", "eval")
            self._code[var_name] = code
        return code

    def sample_equation_parameters(self, var_name: str) -> dict:
        """Sample a value for each local ``#k`` parameter of an equation, keyed ``'#k'``.

        Named parameters (``#name``) are shared across equations and sampled by the model
        (see ``SDM.sample_model_parameters``), not here.
        """
        if var_name not in self.equations:
            return {}
        param_labels = [f'#{idx}' for idx in self.equations[var_name]['parameter_indices']]
        if not param_labels:
            return {}
        sampled = np.random.uniform(self.parameter_range[0], self.parameter_range[1], size=len(param_labels))
        return {label: val for label, val in zip(param_labels, sampled)}

    def evaluate(self, var_name: str, vals: Dict[str, float], params, intervention: float = 0.0,
                 t: float = 0.0) -> float:
        """Evaluate a custom equation.

        Args:
            var_name: name of the variable
            vals: current values of the model variables
            params: parameter values of this equation, ``{'#1': ..., '#rate': ...}``
            intervention: intervention intensity for this variable (default 0.0)
            t: simulation time (default 0.0)

        If the equation contains ``$`` the intervention enters there; otherwise it is added to
        the result.
        """
        if var_name not in self.equations:
            raise ValueError(f"No equation defined for variable '{var_name}'")
        eq_info = self.equations[var_name]
        code = self._compiled(var_name)
        local_ns = {'vals': vals, 'params': params, 'intervention': intervention, 't': t}
        try:
            result = float(eval(code, self._globals, local_ns))
        except KeyError as exc:
            key = exc.args[0] if exc.args else exc
            if isinstance(key, str) and key.startswith('#'):
                raise ValueError(
                    f"Equation for '{var_name}' uses parameter '{key}', but no value was given for it "
                    f"(expected in params['{var_name}']['__eq_params_{var_name}__']).\n"
                    f"Equation: {eq_info['equation']}") from exc
            raise ValueError(f"Error evaluating equation for '{var_name}': no value for variable {key!r}.\n"
                             f"Equation: {eq_info['equation']}") from exc
        except Exception as exc:
            raise ValueError(f"Error evaluating equation for '{var_name}': {exc}\nEquation: {eq_info['equation']}"
                             f"\nCompiled: {eq_info.get('compiled_equation')}") from exc
        # If no $ in equation, add intervention to the result as default behavior
        if not eq_info.get('has_intervention', False) and intervention != 0.0:
            result = result + intervention
        return result

    def has_custom_equation(self, var_name: str) -> bool:
        """Check if a variable has a valid custom equation."""
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
    Parse equations from an Elements dataframe (helper for programmatic use).

    Args:
        df_elements: DataFrame with columns including 'Label' and 'Equation'
        variables: List of cleaned variable names
        adjacency_matrix: DataFrame with adjacency matrix
        variable_list: Ordered list of all variables

    Returns:
        Tuple of (equations_dict, all_warnings). Invalid equations are skipped; their messages
        are included in the warnings.
    """
    parser = EquationParser(variables)
    equations = {}
    all_warnings = []

    if 'Equation' not in df_elements.columns:
        return equations, all_warnings

    for _, row in df_elements.iterrows():
        label = row['Label']
        equation = row.get('Equation', None)
        if pd.isna(equation) or str(equation).strip() == '':
            continue
        var_name = None
        for v in variables:
            if v == label or normalize_name(label) == v:
                var_name = v
                break
        if var_name is None:
            continue
        if var_name in adjacency_matrix.index:
            incoming_links = [col for col in adjacency_matrix.columns
                              if adjacency_matrix.loc[var_name, col] != 0]
        else:
            incoming_links = []
        parsed = parser.parse_equation(var_name, equation, incoming_links)
        if parsed['is_valid']:
            equations[var_name] = parsed
        for warning in parsed['validation_warnings']:
            warning_msg = f"[{var_name}] {warning}"
            all_warnings.append(warning_msg)
            warnings.warn(warning_msg)

    return equations, all_warnings

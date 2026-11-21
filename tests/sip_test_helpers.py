"""Shared helpers for the SIP test suite.

Tests build small Kumu-style workbooks in a temporary folder, extract them and simulate them.
These helpers keep that boilerplate in one place::

    from sip_test_helpers import write_workbook, load_model, quiet, NAN

    path = write_workbook(tmp_path / "m.xlsx",
                          elements={"Label": ["S", "K"], "Type": ["stock", "constant"],
                                    "Tags": [0, 1], "Description": ["VOI", ""],
                                    "Equation": ["2 * K", NAN]},
                          connections=[("K", "+", "S")])
    s, sdm = load_model(path, N=2, t_end=10.0)

``elements`` is a dict of columns (or a DataFrame) for the Elements sheet; ``connections`` a list
of ``(From, Type, To)`` tuples; ``interactions`` a list of ``(From1, From2, Type, To)``;
``parameters`` a list of ``(Name, Low, High)`` rows for the optional Parameters sheet.
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from sip_systemsinsightpipeline import SDM, Extract

NAN = np.nan


def write_workbook(path, elements, connections, interactions=None, parameters=None):
    """Write an Elements/Connections(/Interactions/Parameters) workbook and return its path."""
    df_e = elements if isinstance(elements, pd.DataFrame) else pd.DataFrame(elements)
    with pd.ExcelWriter(path) as xw:
        df_e.to_excel(xw, sheet_name="Elements", index=False)
        pd.DataFrame(list(connections), columns=["From", "Type", "To"]).to_excel(
            xw, sheet_name="Connections", index=False)
        if interactions is not None:
            pd.DataFrame(list(interactions), columns=["From1", "From2", "Type", "To"]).to_excel(
                xw, sheet_name="Interactions", index=False)
        if parameters is not None:
            pd.DataFrame(list(parameters), columns=["Name", "Low", "High"]).to_excel(
                xw, sheet_name="Parameters", index=False)
    return str(path)


def quiet(fn, *args, **kwargs):
    """Call ``fn`` with stdout captured and all warnings recorded.

    Returns ``(result, stdout_text, [warning messages])``.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, out.getvalue(), [str(w.message) for w in caught]


def extract_settings(path, N=1, t_end=10.0, pv=1.0, seed=1, time_unit="years"):
    """Extract a workbook quietly and set the usual simulation settings."""
    s, _, _ = quiet(Extract(str(path)).extract_settings)
    s.seed, s.N, s.t_end, s.time_unit = seed, N, t_end, time_unit
    s.parameter_value_aux = s.parameter_value_stocks = pv
    return s


def load_model(path, N=1, t_end=10.0, pv=1.0, seed=1, time_unit="years"):
    """``(settings, SDM)`` for a workbook, created quietly."""
    s = extract_settings(path, N=N, t_end=t_end, pv=pv, seed=seed, time_unit=time_unit)
    sdm, _, _ = quiet(SDM, s)
    return s, sdm


def fixed_params(sdm, stock_links=1.0, aux_links=0.0, equation_parameters=None):
    """A deterministic ``params`` dict: every link into a stock without equation gets
    ``stock_links``, every other link ``aux_links``; ``equation_parameters`` maps names as in
    ``run_GSA`` (``"Var | #1"``, ``"#name"``) to values."""
    equation_parameters = equation_parameters or {}
    params = {}
    for var in sdm.stocks_and_auxiliaries:
        params[var] = {"Intercept": 0}
        for src in sdm.df_adj.columns:
            if sdm.df_adj.loc[var, src] != 0:
                params[var][src] = stock_links if (var in sdm.stocks and var not in sdm.equations) else aux_links
        if var in sdm.equations:
            eq = {}
            for idx in sdm.equations[var]["parameter_indices"]:
                key = f"{var} | #{idx}"
                if key in equation_parameters:
                    eq[f"#{idx}"] = float(equation_parameters[key])
            for name in sdm.equations[var].get("named_parameters", []):
                if f"#{name}" in equation_parameters:
                    eq[f"#{name}"] = float(equation_parameters[f"#{name}"])
            params[var][f"__eq_params_{var}__"] = eq
    return params

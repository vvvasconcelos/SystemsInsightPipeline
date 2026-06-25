"""Regression guard for the custom-`Equation` column being ignored on the linear SDM path.

A model that defines a node relationship via the Elements `Equation` column but has NO interaction
terms takes the linear SDM path, which builds A/b from per-link coefficients only and never evaluates
the equations -> all-zero dynamics, silently.

`test_equation_column_drives_linear_dynamics` is marked xfail(strict=True): it documents the intended
behaviour and is expected to FAIL until the bug is fixed. When the fix lands it will XPASS, and strict
mode turns an unexpected pass into a failure -- prompting removal of the xfail marker.

Full diagnosis and suggested fix: bug_reports/equation_linear_path.md (+ ..._repro.py, ..._FIX.md).
"""
import pandas as pd
import pytest

from sip_systemsinsightpipeline import SDM, Extract


def write_workbook(path, use_equation):
    """One stock S driven by one constant lever; intended dynamics  dS/dt = 2.0 * Lever.

    use_equation=True  -> express it in the Elements `Equation` column ("2.0 * Lever")
    use_equation=False -> express the SAME structure as an ordinary signed link (no Equation)
    """
    df_e = pd.DataFrame({
        "Label": ["S", "Lever"],
        "Type": ["stock", "constant"],
        "Tags": [0, 1],                                  # Tags=1 -> Lever is an intervention
        "Description": ["VOI", ""],
        "Equation": ["2.0 * Lever" if use_equation else None, None],
    })
    df_c = pd.DataFrame({"From": ["Lever"], "Type": ["+"], "To": ["S"]})
    with pd.ExcelWriter(path) as writer:
        df_e.to_excel(writer, sheet_name="Elements", index=False)
        df_c.to_excel(writer, sheet_name="Connections", index=False)


def build(path, use_equation, force_nonlinear=False):
    write_workbook(path, use_equation)
    s = Extract(str(path)).extract_settings()
    s.seed = 0
    s.N = 1
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_aux = 0.3
    s.parameter_value_stocks = 0.3
    sdm = SDM(s)
    if force_nonlinear:
        # Route the SAME model through the nonlinear ODE path (dz_dt_int), which evaluates equations.
        sdm.interaction_terms = [("__force_nonlinear__",)]
    return s, sdm


def end_state(s, sdm):
    df_sol, _ = sdm.run_simulations()
    voi = s.variable_of_interest[0]
    return float(df_sol[0][0][voi].iloc[-1])


# With dS/dt = 2.0 * Lever, a Lever intervention of 1 over t_end=10 gives S(end) = 20.
EXPECTED_END_STATE = 20.0


def test_equation_column_drives_linear_dynamics(tmp_path):
    """The `Equation` column must drive the dynamics even with no interaction terms.

    Regression guard for the bug where equation-only models (no interaction terms) took the
    linear A/b path and silently ignored the equations; see bug_reports/equation_linear_path.md.
    """
    s, sdm = build(tmp_path / "eq.xlsx", use_equation=True)
    assert end_state(s, sdm) == pytest.approx(EXPECTED_END_STATE, rel=1e-3)


def test_signed_link_baseline_runs(tmp_path):
    """Sanity: the same structure as an ordinary signed link produces non-zero dynamics."""
    s, sdm = build(tmp_path / "link.xlsx", use_equation=False)
    assert end_state(s, sdm) > 0.0


def test_equation_honoured_on_nonlinear_path(tmp_path):
    """The equation itself is correct: forcing the nonlinear path integrates 2.0*Lever -> 20."""
    s, sdm = build(tmp_path / "eq.xlsx", use_equation=True, force_nonlinear=True)
    assert end_state(s, sdm) == pytest.approx(EXPECTED_END_STATE, rel=1e-3)

"""
Minimal, self-contained reproduction of a SIP bug.

BUG: For a model that uses the Elements `Equation` column but has NO interaction terms,
SIP silently ignores the equations and runs all-zero dynamics. The custom-equation values
are only ever evaluated on the *nonlinear* ODE path (`SDM.dz_dt_int` -> `_compute_equation_value`),
which `SDM.run_SDM` invokes only when `self.interaction_terms` is non-empty. With no interaction
terms the *linear* path is used (`params_to_A_b` + `solve_sdm_linear`/`analytical_solution`), and
that path builds the A/b system purely from per-link coefficients -- which `run_simulations`
has just stripped for every custom-equation variable. Result: the affected rows of A and b are
zero, so the variable never changes. No warning is raised.

Tested with sip_systemsinsightpipeline 0.2.1.
Run:  python equation_linear_path_repro.py
"""
import os
import tempfile
import numpy as np
import openpyxl

from sip_systemsinsightpipeline.cld import Extract
from sip_systemsinsightpipeline.sdm import SDM


def make_model(path, use_equation):
    """One stock S driven by one constant lever; intended dynamics  dS/dt = 2.0 * Lever.

    `use_equation=True`  -> express it via the Elements `Equation` column ("2.0 * Lever").
    `use_equation=False` -> express the SAME structure as an ordinary signed link (no Equation).
    """
    wb = openpyxl.Workbook()
    e = wb.active
    e.title = "Elements"
    e.append(["Label", "Type", "Tags", "Description", "Equation"])
    e.append(["S", "Stock", 0, "VOI", "2.0 * Lever" if use_equation else None])
    e.append(["Lever", "Constant", 1, None, None])          # Tags=1 -> intervention lever
    c = wb.create_sheet("Connections")
    c.append(["From", "To", "Direction", "Label", "Type", "Tags", "Description"])
    c.append(["Lever", "S", "directed", None, "+", None, None])
    wb.save(path)


def make_sdm(path, force_nonlinear=False):
    s = Extract(path).extract_settings()
    s.seed = 0
    s.N = 1
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_aux = 0.3
    s.parameter_value_stocks = 0.3
    sdm = SDM(s)
    if force_nonlinear:
        # Route the SAME model through the nonlinear ODE path (dz_dt_int), which evaluates
        # custom equations. This is only to demonstrate that the equation itself is fine.
        sdm.interaction_terms = [("__force_nonlinear__",)]
    return s, sdm


def end_state_and_effect(s, sdm):
    df_sol, _ = sdm.run_simulations()
    voi = s.variable_of_interest[0]
    eff = sdm.get_intervention_effects()
    return float(df_sol[0][0][voi].iloc[-1]), float(np.median(eff[voi]["Lever"]))


if __name__ == "__main__":
    tmp = tempfile.gettempdir()
    p_eq = os.path.join(tmp, "sip_repro_equation.xlsx")
    p_link = os.path.join(tmp, "sip_repro_link.xlsx")
    make_model(p_eq, use_equation=True)
    make_model(p_link, use_equation=False)

    print("=" * 78)
    print("Model: 1 Stock S, 1 Constant lever 'Lever'.  Intended dS/dt = 2.0 * Lever.")
    print("Settings: t_end=10, Lever intervention strength = 1  ->  correct S(end) = 20.0")
    print("=" * 78)

    s, sdm = make_sdm(p_eq)
    end, eff = end_state_and_effect(s, sdm)
    print(f"\nA) Equation column, default (linear) path : S(end)={end:7.3f}  effect={eff:7.3f}   <-- BUG (should be ~20)")

    s, sdm = make_sdm(p_link)
    end, eff = end_state_and_effect(s, sdm)
    print(f"B) Same structure as a signed link        : S(end)={end:7.3f}  effect={eff:7.3f}   (works; non-zero)")

    s, sdm = make_sdm(p_eq, force_nonlinear=True)
    end, eff = end_state_and_effect(s, sdm)
    print(f"C) Equation column, forced nonlinear path : S(end)={end:7.3f}  effect={eff:7.3f}   (equation IS honoured)")

    # ---- Root-cause evidence: inspect the linear system A, b for case A ----
    print("\n" + "-" * 78)
    print("Root cause (case A): the linear system built by params_to_A_b is identically zero")
    print("-" * 78)
    s, sdm = make_sdm(p_eq)
    params = sdm.sample_model_parameters()
    # Reproduce the 'PATCH' in run_simulations that strips link params for custom-equation vars:
    for var in params:
        if sdm._equation_evaluator and sdm._equation_evaluator.has_custom_equation(var):
            keep = [k for k in params[var] if k.startswith("__eq_params_") or k == "Intercept"]
            params[var] = {k: params[var][k] for k in keep}
    cv = np.zeros(len(sdm.constants))
    cv[sdm.constants.index("Lever")] = 1.0          # Lever intervention = 1
    A, b = sdm.params_to_A_b(params, cv)
    print("params['S'] after the strip :", params["S"], " (the 'Lever' coefficient is gone)")
    print("A =", A.tolist(), "  b =", b.tolist(), "  ->  dS/dt = 0  for all t")

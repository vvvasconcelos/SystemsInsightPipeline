"""Tests for the simulation core (sip_systemsinsightpipeline.sdm.SDM)."""
import numpy as np
import pytest

from sip_systemsinsightpipeline import IntegrationError, LoopsThatMatter
from sip_test_helpers import NAN, fixed_params, load_model, quiet, write_workbook


def decay_model(tmp_path, name="decay.xlsx"):
    """dS/dt = K - 0.5 S (custom equation); exact solution S(t) = 2 K (1 - exp(-t/2))."""
    return write_workbook(tmp_path / name,
                          {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                           "Description": ["VOI", ""], "Equation": ["K - 0.5 * S", NAN]},
                          [("K", "+", "S")])


def exact_decay(t, k=1.0):
    return 2 * k * (1 - np.exp(-t / 2))


def linear_cld(tmp_path):
    return write_workbook(tmp_path / "cld.xlsx",
                          {"Label": ["S", "A", "L1", "L2"], "Type": ["stock", "auxiliary", "constant", "constant"],
                           "Tags": [0, 0, 1, -1], "Description": ["VOI", "", "", ""]},
                          [("A", "+", "S"), ("S", "-", "S"), ("L1", "+", "A"), ("L2", "+", "A")])


# ----------------------------------------------------------------------------- intervention effects (I-06)
def test_cld_baseline_is_zero_so_effects_equal_final_values(tmp_path):
    s, sdm = load_model(linear_cld(tmp_path), N=4, pv=0.3)
    quiet(sdm.run_simulations)
    for base in sdm.df_sol_baseline_per_sample:
        assert np.allclose(base[["S", "A"]].to_numpy(), 0.0)
    rel = sdm.get_intervention_effects()["S"]
    raw = sdm.get_intervention_effects(relative_to=None)["S"]
    for lever in ("L1", "L2"):
        assert np.allclose(rel[lever], raw[lever])
    assert np.all(np.asarray(rel["L1"]) > 0) and np.all(np.asarray(rel["L2"]) < 0)


def test_effects_are_sorted_by_median_absolute_effect(tmp_path):
    wb = write_workbook(tmp_path / "two.xlsx",
                        {"Label": ["S", "Small", "Big"], "Type": ["stock", "constant", "constant"],
                         "Tags": [0, 1, 1], "Description": ["VOI", "", ""], "Equation": ["Small + 5 * Big + 1", NAN, NAN]},
                        [("Small", "+", "S"), ("Big", "+", "S")])
    s, sdm = load_model(wb, N=2, t_end=1.0)
    quiet(sdm.run_simulations)
    effects = sdm.get_intervention_effects()["S"]
    assert list(effects) == ["Big", "Small"]
    assert np.allclose(effects["Big"], 5.0) and np.allclose(effects["Small"], 1.0)


def test_get_intervention_effects_argument_checks(tmp_path):
    s, sdm = load_model(decay_model(tmp_path))
    with pytest.raises(ValueError, match="run_simulations"):
        sdm.get_intervention_effects()
    quiet(sdm.run_simulations)
    with pytest.raises(ValueError, match="relative_to"):
        sdm.get_intervention_effects(relative_to="zero")


def test_compare_interventions_table_and_run_sa_run(tmp_path):
    s, sdm = load_model(linear_cld(tmp_path), N=12, pv=0.3)
    quiet(sdm.run_simulations)
    effects = sdm.get_intervention_effects()["S"]
    _, out, _ = quiet(sdm.compare_interventions_table, effects, n_bootstraps=20)
    assert "Cliff" in out
    (rho, df_SA), _, _ = quiet(sdm.run_SA, "S", None, n_bootstraps=20)
    assert len(df_SA) == 12 * 2 and "Effect" in df_SA


# ----------------------------------------------------------------------------- time in equations (I-07)
def test_time_in_auxiliary_equation_is_evaluated_per_time_point(tmp_path):
    wb = write_workbook(tmp_path / "ramp.xlsx",
                        {"Label": ["S", "Ramp", "K"], "Type": ["stock", "auxiliary", "constant"],
                         "Tags": [0, 0, 1], "Description": ["VOI", "", ""], "Equation": ["Ramp", "K * t", NAN]},
                        [("Ramp", "+", "S"), ("K", "+", "Ramp")])
    s, sdm = load_model(wb, t_end=4.0)
    sdm.t_eval = np.linspace(0, 4, 5)
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))
    assert np.allclose(df["Ramp"], df["Time"])                    # Ramp = K t with K = 1
    assert df["S"].iloc[-1] == pytest.approx(4.0 ** 2 / 2, rel=1e-5)


# ----------------------------------------------------------------------------- solvers (I-14)
@pytest.mark.parametrize("solver, dt, tol", [("LSODA", None, 1e-5), ("BDF", None, 1e-4), ("RK45", None, 1e-5),
                                             ("RK4", 0.01, 1e-8), ("Euler", 0.001, 1e-3)])
def test_solvers_reproduce_exact_solution(tmp_path, solver, dt, tol):
    s, sdm = load_model(decay_model(tmp_path), t_end=6.0)
    sdm.solver, sdm.solver_options["dt"] = solver, dt
    sdm.t_eval = np.array([0.0, 1.0, 2.5, 6.0])                    # dt does not divide 2.5 - 1.0
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))
    assert np.allclose(df["S"].to_numpy(), exact_decay(sdm.t_eval), atol=tol)


def test_fixed_step_default_dt(tmp_path):
    s, sdm = load_model(decay_model(tmp_path), t_end=6.0)
    sdm.solver = "rk4"                                            # case-insensitive
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))
    assert df["S"].iloc[-1] == pytest.approx(exact_decay(6.0), abs=1e-8)


def test_rhs_budget_raises_clear_error(tmp_path):
    s, sdm = load_model(decay_model(tmp_path))
    sdm.solver_options["max_rhs_evals"] = 5
    with pytest.raises(IntegrationError, match="right-hand-side evaluations"):
        quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))
    sdm.solver_options["max_rhs_evals"] = None                     # guard disabled
    quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))


def test_unknown_solver_option_is_rejected(tmp_path):
    s, sdm = load_model(decay_model(tmp_path))
    sdm.solver_options["max_steps"] = 10
    with pytest.raises(ValueError, match="Unknown solver option"):
        quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))


def test_failed_integration_raises_instead_of_returning_partial_results(tmp_path):
    wb = write_workbook(tmp_path / "blowup.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["(S + 1) ** 2 + K", NAN]},
                        [("K", "+", "S")])
    s, sdm = load_model(wb, t_end=5.0)                            # finite-time blow-up at t ~ 1
    sdm.large_value_threshold = None
    with pytest.raises(IntegrationError):
        quiet(sdm.run_SDM_with_intervention_intensities, [0.0], fixed_params(sdm))


# ----------------------------------------------------------------------------- auxiliary ordering (I-03)
def test_long_chain_of_auxiliaries_is_sorted(tmp_path):
    n = 150                                                       # 0.5.0 gave up after 100 levels
    labels = [f"A{i}" for i in range(n)]
    eqs = ["K"] + [f"A{i - 1} + 1" for i in range(1, n)]
    wb = write_workbook(tmp_path / "chain.xlsx",
                        {"Label": labels[::-1] + ["K"], "Type": ["auxiliary"] * n + ["constant"],
                         "Tags": [0] * n + [1], "Description": ["VOI"] + [""] * n, "Equation": eqs[::-1] + [NAN]},
                        [("K", "+", "A0")] + [(f"A{i - 1}", "+", f"A{i}") for i in range(1, n)])
    s, sdm = load_model(wb, t_end=1.0)
    sdm.large_value_threshold = None
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))
    assert df[f"A{n - 1}"].iloc[-1] == pytest.approx(1.0 + (n - 1))


def test_loop_of_auxiliaries_in_the_diagram_is_rejected_at_extraction(tmp_path):
    wb = write_workbook(tmp_path / "loop.xlsx",
                        {"Label": ["S", "A", "B", "K"], "Type": ["stock", "auxiliary", "auxiliary", "constant"],
                         "Tags": [0, 0, 0, 1], "Description": ["VOI", "", "", ""],
                         "Equation": ["A", "B + K", "0.5 * A", NAN]},
                        [("A", "+", "S"), ("B", "+", "A"), ("K", "+", "A"), ("A", "+", "B"), ("S", "+", "B")])
    with pytest.raises(Exception, match="at least one stock"):
        load_model(wb)


def test_algebraic_loop_in_equations_is_reported_with_its_members(tmp_path):
    """A and B reference each other only in their equations (not in the diagram)."""
    wb = write_workbook(tmp_path / "loop2.xlsx",
                        {"Label": ["S", "A", "B", "K"], "Type": ["stock", "auxiliary", "auxiliary", "constant"],
                         "Tags": [0, 0, 0, 1], "Description": ["VOI", "", "", ""],
                         "Equation": ["A", "B + K", "0.5 * A", NAN]},
                        [("A", "+", "S"), ("K", "+", "A"), ("S", "+", "B")])
    s, sdm = load_model(wb)
    with pytest.raises(ValueError, match=r"A -> B -> A"):
        quiet(sdm.run_SDM_with_intervention_intensities, [1.0], fixed_params(sdm))


def test_interaction_parents_are_dependencies(tmp_path):
    """R depends on Q only through an interaction term; Q must be computed before R."""
    wb = write_workbook(tmp_path / "inter.xlsx",
                        {"Label": ["R", "Q", "P", "S"], "Type": ["auxiliary", "auxiliary", "constant", "stock"],
                         "Tags": [0, 0, 1, 0], "Description": ["", "", "", "VOI"]},
                        [("P", "+", "R"), ("P", "+", "Q"), ("R", "+", "S")],
                        interactions=[("P", "Q", "+", "R")])
    s, sdm = load_model(wb, t_end=1.0, pv=0.5)
    params = sdm.sample_model_parameters()
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], params)
    expected_q = params["Q"]["P"] * 1.0
    expected_r = params["R"]["P"] * 1.0 + params["R"]["Q * P"] * expected_q * 1.0
    assert df["Q"].iloc[-1] == pytest.approx(expected_q)
    assert df["R"].iloc[-1] == pytest.approx(expected_r)


# ----------------------------------------------------------------------------- state hygiene
def test_no_intervention_leaks_after_run_simulations(tmp_path):
    wb = write_workbook(tmp_path / "leak.xlsx",
                        {"Label": ["S", "A", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 1, 0],
                         "Description": ["VOI", "", ""], "Equation": ["A", "2 * K", NAN]},
                        [("A", "+", "S"), ("K", "+", "A")])
    s, sdm = load_model(wb, t_end=1.0)
    quiet(sdm.run_simulations)                                    # intervenes on auxiliary A
    assert sdm._current_intervention_intensities == {}
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    (df, _, _), _, _ = quiet(ltm.run_ltm_analysis, fixed_params(sdm), n_points=5, progress_bar=False)
    assert np.allclose(df["A"], 0.0) and np.allclose(df["S"], 0.0)


def test_direct_run_sdm_reports_large_values_once_per_call(tmp_path):
    s, sdm = load_model(decay_model(tmp_path), t_end=1.0)
    sdm.large_value_threshold = 0.1
    x0 = np.zeros(1)
    params = fixed_params(sdm)
    _, _, warns = quiet(sdm.run_SDM, x0, np.array([1.0]), None, None, params)
    assert sum("larger than" in w for w in warns) == 1

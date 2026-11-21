"""The public stock-and-flow tutorial model, ``tutorials/StockFlow.xlsx``.

StockFlow is a small synthetic model (trainees become certified installers, who retrofit homes)
written the way applied stock-and-flow models are written in SIP custom equations; see
``tutorials/StockFlow.ipynb``. Its golden values are checked in ``test_tutorial_models.py``.
These tests check the features it is meant to exercise: stocks that integrate their net flows, a
transfer whose rate is a named parameter shared by two equations with its range in the Parameters
sheet, a local parameter, a policy that ramps in with time ``t`` (a non-zero baseline), whole-name
matching of names with punctuation, balancing and reinforcing loops, fixed-step integration, and
the analyses that sample the parameters (``run_GSA``, ``sample_outcomes``).
"""
import numpy as np
import pytest

from sip_systemsinsightpipeline import LoopsThatMatter
from sip_test_helpers import load_model, quiet

TRAINEES, INSTALLERS, VOI = "Trainees", "Certified installers (FTE)", "Homes retrofitted"
RETROFITS, PROGRAMME = "Homes retrofitted per year", "National programme (phase-in)"
LEVERS = ["Training subsidy", "Awareness campaign", "Retention bonus"]
PRIORS = {"completion_rate": (0.4, 0.8), "enrol_rate": (0.1, 0.3), "attrition_rate": (0.05, 0.15),
          "productivity": (0.02, 0.08), "word_of_mouth": (0.5, 1.5), "forgetting": (0.25, 0.5)}
PVA = 0.2  # parameter_value_aux: range [0, PVA] of the local parameter #1


def stockflow(tutorials_dir, N=2):
    return load_model(tutorials_dir / "StockFlow.xlsx", N=N, t_end=20.0, pv=PVA, seed=12345)


def test_structure(tutorials_dir):
    s, sdm = stockflow(tutorials_dir)
    assert s.stocks == [TRAINEES, INSTALLERS, VOI, "Public awareness"]
    assert set(s.stocks) <= set(s.equations)                  # every stock integrates its net flow
    assert s.intervention_variables == LEVERS
    assert all(s.var_to_type[v] == "constant" for v in LEVERS)
    assert s.variable_of_interest == [VOI]
    assert s.equation_warnings == []                          # equations match the diagram
    assert s.parameter_priors == PRIORS
    # the transfer rate is one parameter shared by the two equations it connects
    assert sdm._named_param_users["completion_rate"] == [TRAINEES, INSTALLERS]
    assert sdm.named_parameters == sorted(PRIORS)
    assert s.equations["Enrolment"]["parameter_indices"] == [1]
    assert s.equations[PROGRAMME]["uses_time"] and s.equations[PROGRAMME]["variables_used"] == set()
    # whole names: 'Homes retrofitted' is not found inside 'Homes retrofitted per year'
    assert s.equations[VOI]["variables_used"] == {RETROFITS}
    assert s.equations[RETROFITS]["variables_used"] == {INSTALLERS, VOI}


def test_loops(tutorials_dir):
    s, sdm = stockflow(tutorials_dir)
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    assert dict(zip(ltm.loop_names, ltm.feedback_loops)) == {
        "B1": [INSTALLERS, "Attrition"],                                    # attrition
        "B2": [VOI, RETROFITS],                                             # market saturation
        "R1": [TRAINEES, INSTALLERS, RETROFITS, "Public awareness", "Enrolment"],  # visible retrofits attract trainees
    }


def test_transfer_moves_people_between_the_two_stocks(tutorials_dir):
    """What leaves Trainees enters Certified installers: with the shared rate the transfer
    cancels in the sum, d(Trainees + Installers)/dt = Enrolment - Attrition, in every state."""
    s, sdm = stockflow(tutorials_dir)
    np.random.seed(3)
    params = sdm.sample_model_parameters()
    rate = {v: params[v][f"__eq_params_{v}__"]["#completion_rate"] for v in (TRAINEES, INSTALLERS)}
    assert rate[TRAINEES] == rate[INSTALLERS]
    assert PRIORS["completion_rate"][0] <= rate[TRAINEES] <= PRIORS["completion_rate"][1]
    i, j = s.stocks.index(TRAINEES), s.stocks.index(INSTALLERS)
    rng = np.random.default_rng(0)
    for _ in range(5):
        z, consts, t = rng.uniform(0, 1, len(s.stocks)), rng.uniform(0, 1, len(s.constants)), rng.uniform(0, 20)
        dz = sdm.dz_dt_int(t, z, consts, params)
        vals = sdm._compute_all_auxiliaries(z, consts, params, t)
        assert dz[i] + dz[j] == pytest.approx(vals["Enrolment"] - vals["Attrition"], abs=1e-12)
        assert dz[i] < vals["Enrolment"]                      # the transfer drains Trainees


def test_programme_ramps_in_and_effects_are_measured_against_the_baseline(tutorials_dir):
    s, sdm = stockflow(tutorials_dir, N=2)
    sdm.t_eval = np.linspace(0.0, 20.0, 41)
    quiet(sdm.run_simulations)
    for base in sdm.df_sol_baseline_per_sample:
        t = base.index.to_numpy()
        np.testing.assert_allclose(base[PROGRAMME], np.clip((t - 2) / 3, 0, 1), atol=1e-12)
        # nothing happens before the programme starts (up to the solver's interpolation error)
        assert np.abs(base.loc[base.index <= 2.0, s.stocks].to_numpy()).max() < sdm.solver_options["atol"]
        assert base[VOI].iloc[-1] > 0.1                       # the baseline is not zero ...
    effects, _, _ = quiet(sdm.get_intervention_effects)
    raw, _, _ = quiet(sdm.get_intervention_effects, relative_to=None)
    for lever in LEVERS:
        for n, base in enumerate(sdm.df_sol_baseline_per_sample):
            # ... so the effect is the change relative to it, not the final value
            assert effects[VOI][lever][n] == pytest.approx(raw[VOI][lever][n] - base[VOI].iloc[-1], abs=1e-12)
            assert 0.0 < effects[VOI][lever][n] < raw[VOI][lever][n]


def test_fixed_step_rk4_agrees_with_lsoda(tutorials_dir):
    s, sdm = stockflow(tutorials_dir)
    np.random.seed(5)
    params = sdm.sample_model_parameters()
    sdm.t_eval = np.linspace(0.0, 20.0, 11)
    lsoda, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, np.ones(len(LEVERS)), params)
    sdm.solver, sdm.solver_options["dt"] = "RK4", 0.01
    rk4, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, np.ones(len(LEVERS)), params)
    np.testing.assert_allclose(rk4[s.stocks].to_numpy(), lsoda[s.stocks].to_numpy(), rtol=1e-4, atol=1e-5)


def test_gsa_and_sample_outcomes_honour_the_parameter_ranges(tutorials_dir):
    s, sdm = stockflow(tutorials_dir)
    names = ["Enrolment | #1"] + [f"#{p}" for p in sorted(PRIORS)]
    df, _, _ = quiet(sdm.run_GSA, method="delta", n=128, seed=1, n_bootstrap=20, show_progress=False)
    assert sorted(df["parameter"]) == sorted(names)           # equation parameters only, each once
    assert (df["delta"] >= 0).all()
    assert df.loc[df["delta"].idxmax(), "parameter"] == "#productivity"
    X, y = quiet(sdm.sample_outcomes, n=20, seed=2, relative_to="baseline", show_progress=False)[0]
    assert list(X.columns) == list(sdm._gsa_design.columns) and len(y) == 20
    for design in (sdm._gsa_design, X):
        for p, (low, high) in PRIORS.items():
            assert design[f"#{p}"].between(low, high).all()
        assert design["Enrolment | #1"].between(0.0, PVA).all()
    assert np.isfinite(y).all() and (y > 0).all()             # the full package always adds retrofits

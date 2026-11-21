"""Named equation parameters (#name) and their ranges from the Parameters sheet."""
import numpy as np
import pytest

from sip_systemsinsightpipeline import LoopsThatMatter, SDMOptimizer
from sip_test_helpers import NAN, load_model, quiet, write_workbook


def model(tmp_path, eq_s1="-#rate * S1 + #1 * K", eq_s2="#rate * S1", parameters=None, name="m.xlsx"):
    """Two stocks: what flows out of S1 at rate #rate flows into S2; #1 is local to S1."""
    return write_workbook(
        tmp_path / name,
        {"Label": ["S1", "S2", "K", "A"], "Type": ["stock", "stock", "constant", "auxiliary"],
         "Tags": [0, 0, 1, 0], "Description": ["VOI", "", "", ""],
         "Equation": [eq_s1, eq_s2, NAN, NAN]},
        [("K", "+", "S1"), ("S1", "+", "S2"), ("S2", "+", "A"), ("A", "-", "S1")],
        parameters=parameters)


def test_default_range_is_zero_to_parameter_value_aux(tmp_path):
    s, sdm = load_model(model(tmp_path), pv=0.4)
    assert sdm.named_parameters == ["rate"]
    assert sdm.named_parameter_bounds("rate") == (0.0, 0.4)
    draws = [sdm.sample_model_parameters()["S2"]["__eq_params_S2__"]["#rate"] for _ in range(200)]
    assert 0.0 <= min(draws) and max(draws) <= 0.4 and np.std(draws) > 0.05
    sdm.parameter_value_aux = 2.0                       # the default follows the current setting
    assert sdm.named_parameter_bounds("rate") == (0.0, 2.0)


def test_range_from_parameters_sheet_and_fixed_values(tmp_path):
    s, sdm = load_model(model(tmp_path, eq_s1="-#rate * S1 + #gain * K",
                              parameters=[("#rate", 0.1, 0.1), ("gain", -2.0, -1.0)]))
    assert sdm.parameter_priors == {"rate": (0.1, 0.1), "gain": (-2.0, -1.0)}
    for _ in range(20):
        p = sdm.sample_model_parameters()["S1"]["__eq_params_S1__"]
        assert p["#rate"] == 0.1 and -2.0 <= p["#gain"] <= -1.0
    free, fixed = sdm._gsa_parameter_spec()
    assert [e["name"] for e in free] == ["A <- S2", "#gain"] and [e["name"] for e in fixed] == ["#rate"]


def test_named_parameters_are_drawn_after_all_other_parameters(tmp_path):
    """Within one draw, the link and local parameters are the same whether or not an equation
    also uses a named parameter (named ones are drawn last; later draws then differ because
    the named draws consume random numbers). Models without named parameters therefore keep
    exactly the random sequence of SIP 0.5."""
    _, a = load_model(model(tmp_path, eq_s2="0.3 * S1", name="a.xlsx"), seed=7)
    _, b = load_model(model(tmp_path, eq_s2="#rate * S1", name="b.xlsx"), seed=7)
    np.random.seed(7)                      # SIP samples from numpy's global random state
    pa = a.sample_model_parameters()
    np.random.seed(7)
    pb = b.sample_model_parameters()
    assert pa["A"] == pb["A"]                                            # link coefficients
    assert pa["S1"]["__eq_params_S1__"]["#1"] == pb["S1"]["__eq_params_S1__"]["#1"]
    assert "#rate" in pb["S2"]["__eq_params_S2__"]


def test_gsa_treats_a_shared_parameter_as_one_input(tmp_path):
    s, sdm = load_model(model(tmp_path), pv=0.5)
    free, _ = sdm._gsa_parameter_spec()
    names = [e["name"] for e in free]
    assert names.count("#rate") == 1 and "S1 | #1" in names and "A <- S2" in names
    params = sdm._gsa_params_from_vector([0.3] * len(free), free, [])
    assert params["S1"]["__eq_params_S1__"]["#rate"] == params["S2"]["__eq_params_S2__"]["#rate"] == 0.3
    X, y = quiet(sdm.sample_outcomes, n=20, seed=3, show_progress=False, bounds={"#rate": (0.2, 0.25)})[0]
    assert X["#rate"].between(0.2, 0.25).all() and np.isfinite(y).all()
    df = quiet(sdm.run_GSA, method="delta", n=200, seed=1, show_progress=False)[0]
    assert set(df["parameter"]) == set(names)


def test_bounds_for_unknown_parameter_warn(tmp_path):
    s, sdm = load_model(model(tmp_path))
    _, _, warns = quiet(sdm._gsa_parameter_spec, {"#nope": (0, 1)})
    assert any("unknown parameter" in w for w in warns)


def test_run_sa_reports_named_parameter_once(tmp_path):
    s, sdm = load_model(model(tmp_path), N=8, pv=0.5)
    quiet(sdm.run_simulations)
    (_, df_SA), _, _ = quiet(sdm.run_SA, "S1", None, n_bootstraps=10)
    assert list(df_SA.columns).count("#rate") == 1 and "p1->S1" in df_SA.columns


def test_optimizer_and_ltm_use_the_shared_draw(tmp_path):
    s, sdm = load_model(model(tmp_path), N=2, pv=0.5)
    df = quiet(SDMOptimizer(sdm).optimize_across_parameter_samples, [1.0], "S1", n_starts=1,
               show_progress=False)[0]
    assert (df["param_S1____eq_params_S1____#rate"] == df["param_S2____eq_params_S2____#rate"]).all()
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    out = quiet(ltm.run_ltm_across_samples, n_samples=2, n_points=10, intervention_intensities=[1.0])[0]
    assert len(out) == 2 * 9

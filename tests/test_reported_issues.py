"""Acceptance tests for the issues of the external review of SIP 0.5.0 (November 2026).

One test (or a few) per issue, named ``test_I<nn>_...``. They encode each issue's definition of
done (see ``bug_reports/external_review_2026-11.md``) and failed on 0.5.0 (commit ad14fae).
Most are the reviewers' own regression tests, adapted to the designs agreed for 0.6.0; the
deeper, module-level tests live in ``test_equations.py``, ``test_simulation.py``,
``test_parameters.py``, ``test_ltm.py`` and friends.
"""
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline import (SDM, Extract, IntegrationError, LoopsThatMatter,
                                        SDMOptimizer, discover_scenarios)
from sip_systemsinsightpipeline.equations import EquationEvaluator, EquationParser
from sip_test_helpers import NAN, extract_settings, fixed_params, load_model, quiet, write_workbook

REPO = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- I-01
def test_I01_bare_hash_parameters_compile_and_are_distinct():
    p = EquationParser(["A", "B", "Y"])
    for eq in ("# * A + # * B", "#1 * A + # * B"):
        r = p.parse_equation("Y", eq, ["A", "B"])
        assert r["n_parameters"] == 2, eq
        keys = [f"#{i}" for i in r["parameter_indices"]]
        val = EquationEvaluator({"Y": r}).evaluate("Y", {"A": 1.0, "B": 10.0}, dict(zip(keys, [2.0, 3.0])), 0.0)
        assert val == pytest.approx(2.0 * 1.0 + 3.0 * 10.0), eq


def test_I01_bare_hash_parameters_are_sampled_and_simulated(tmp_path):
    wb = write_workbook(tmp_path / "i01.xlsx",
                        {"Label": ["Y", "A", "B"], "Type": ["auxiliary", "constant", "constant"],
                         "Tags": [0, 1, 1], "Description": ["VOI", "", ""],
                         "Equation": ["# * A + # * B", NAN, NAN]},
                        [("A", "+", "Y"), ("B", "+", "Y")])
    s, sdm = load_model(wb, N=2)
    params = sdm.sample_model_parameters()
    assert set(params["Y"]["__eq_params_Y__"]) == {"#1", "#2"}
    free, _ = sdm._gsa_parameter_spec()
    assert [e["name"] for e in free] == ["Y | #1", "Y | #2"]
    quiet(sdm.run_simulations)
    assert np.isfinite(np.asarray(sdm.get_intervention_effects()["Y"]["A"])).all()


# ----------------------------------------------------------------------------- I-02
def test_I02_unknown_identifier_is_rejected_at_extraction(tmp_path):
    wb = write_workbook(tmp_path / "i02.xlsx",
                        {"Label": ["Habit", "Intention", "Y"], "Type": ["constant", "constant", "auxiliary"],
                         "Tags": [1, 0, 0], "Description": ["", "", "VOI"],
                         "Equation": [NAN, NAN, "#1 * Habit + #2 * Intenton"]},   # typo on purpose
                        [("Habit", "+", "Y"), ("Intention", "+", "Y")])
    with pytest.raises(ValueError, match="Intenton"):
        quiet(Extract(wb).extract_settings)


def test_I02_valid_equation_has_no_unknown_identifier_error():
    r = EquationParser(["Habit", "Intention", "Y"]).parse_equation(
        "Y", "#1 * Habit + #2 * Intention", ["Habit", "Intention"])
    assert r["is_valid"] and r["validation_warnings"] == []


# ----------------------------------------------------------------------------- I-03
def test_I03_substring_names_are_not_reported_as_used():
    r = EquationParser(["Price", "Price index", "Y"]).parse_equation("Y", "#1 * Price index", ["Price index"])
    assert r["variables_used"] == {"Price index"}
    assert r["validation_warnings"] == []


def test_I03_substring_names_do_not_create_circular_dependency(tmp_path):
    wb = write_workbook(tmp_path / "i03.xlsx",
                        {"Label": ["Level", "Level change"], "Type": ["auxiliary", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["1 + Level change", NAN]},
                        [("Level change", "+", "Level")])
    s, sdm = load_model(wb)
    params = sdm.sample_model_parameters()
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], params)
    assert df["Level"].iloc[-1] == pytest.approx(2.0)
    assert sdm.sort_auxiliaries(params) == ["Level"]


# ----------------------------------------------------------------------------- I-04
def test_I04_names_with_parentheses_work(tmp_path):
    V = "Relative price difference (plant vs. animal)"
    wb = write_workbook(tmp_path / "i04.xlsx",
                        {"Label": [V, "Y"], "Type": ["constant", "auxiliary"], "Tags": [1, 0],
                         "Description": ["", "VOI"], "Equation": [NAN, f"0.5 * {V}"]},
                        [(V, "+", "Y")])
    s, sdm = load_model(wb)
    params = sdm.sample_model_parameters()
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], params)
    assert df["Y"].iloc[-1] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- I-05
def test_I05_ltm_evaluates_custom_equation_stocks(tmp_path):
    wb = write_workbook(tmp_path / "i05.xlsx",
                        {"Label": ["S", "A", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                         "Description": ["VOI", "", ""], "Equation": ["0.5 * A - 0.2 * S", "S + K", NAN]},
                        [("A", "+", "S"), ("S", "+", "A"), ("K", "+", "A")])
    s, sdm = load_model(wb)
    params = sdm.sample_model_parameters()
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    (_, links, loops), _, _ = quiet(ltm.run_ltm_analysis, params, n_points=20,
                                    intervention_intensities=[1.0], progress_bar=False)
    vals = {"S": 1.0, "A": 2.0, "K": 1.0}
    assert ltm._eval_equation("S", vals, params) == pytest.approx(0.5 * 2.0 - 0.2 * 1.0)
    assert np.isfinite(loops["R1_raw"]).all()


# ----------------------------------------------------------------------------- I-06
def test_I06_effects_are_baseline_subtracted(tmp_path):
    """dS/dt = 1 + K. Over 10 time units the effect of K = 1 is 10 (the drift adds another 10)."""
    wb = write_workbook(tmp_path / "i06.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["1.0 + K", NAN]},
                        [("K", "+", "S")])
    s, sdm = load_model(wb, N=2, t_end=10.0)
    quiet(sdm.run_simulations)
    assert np.allclose(sdm.get_intervention_effects()["S"]["K"], 10.0, atol=1e-6)
    # the raw end values remain available
    assert np.allclose(sdm.get_intervention_effects(relative_to=None)["S"]["K"], 20.0, atol=1e-6)
    # run_SA measures the same effect
    _, df_SA = quiet(sdm.run_SA, "S", None)[0]
    assert np.allclose(df_SA["Effect"], 10.0, atol=1e-6)


# ----------------------------------------------------------------------------- I-07
def test_I07_time_in_equations(tmp_path):
    wb = write_workbook(tmp_path / "i07.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["K * (t > 2)", NAN]},
                        [("K", "+", "S")])
    s, sdm = load_model(wb, t_end=10.0)
    params = sdm.sample_model_parameters()
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], params)
    assert df["S"].iloc[-1] == pytest.approx(8.0, rel=1e-3)


# ----------------------------------------------------------------------------- I-08
def test_I08_self_contained_equation_without_links(tmp_path):
    """A clock stock (dClock/dt = 1) needs no dummy incoming link."""
    wb = write_workbook(tmp_path / "i08.xlsx",
                        {"Label": ["Clock", "Y", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                         "Description": ["", "VOI", ""], "Equation": ["1", "Clock + K", NAN]},
                        [("Clock", "+", "Y"), ("K", "+", "Y")])
    s, sdm = load_model(wb, t_end=10.0)
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [1.0], sdm.sample_model_parameters())
    assert df["Clock"].iloc[-1] == pytest.approx(10.0)


# ----------------------------------------------------------------------------- I-09 / I-10
def _two_stock_named_model(tmp_path, parameters=None):
    return write_workbook(
        tmp_path / "named.xlsx",
        {"Label": ["S1", "S2", "K"], "Type": ["stock", "stock", "constant"], "Tags": [0, 0, 1],
         "Description": ["VOI", "", ""], "Equation": ["-#rate * S1 + K", "#rate * S1", NAN]},
        [("K", "+", "S1"), ("S1", "+", "S2")], parameters=parameters)


def test_I09_named_parameter_is_shared_across_equations(tmp_path):
    s, sdm = load_model(_two_stock_named_model(tmp_path), N=3)
    for _ in range(3):
        p = sdm.sample_model_parameters()
        assert p["S1"]["__eq_params_S1__"]["#rate"] == p["S2"]["__eq_params_S2__"]["#rate"]
    quiet(sdm.run_simulations)
    for n in range(3):
        df = sdm.df_sol_per_sample[n][0]
        # what leaves S1 enters S2: S1 + S2 integrates K exactly
        assert (df["S1"] + df["S2"]).iloc[-1] == pytest.approx(10.0, rel=1e-5)


def test_I10_one_prior_specification_is_honoured_by_all_samplers(tmp_path):
    s, sdm = load_model(_two_stock_named_model(tmp_path, parameters=[("rate", 0.2, 0.3)]), N=20)
    assert sdm.parameter_priors == {"rate": (0.2, 0.3)}
    draws = [sdm.sample_model_parameters()["S1"]["__eq_params_S1__"]["#rate"] for _ in range(50)]
    assert 0.2 <= min(draws) and max(draws) <= 0.3                  # run_simulations, optimizer, LTM
    free, _ = sdm._gsa_parameter_spec()                              # run_GSA, sample_outcomes
    assert [(e["name"], e["low"], e["high"]) for e in free] == [("#rate", 0.2, 0.3)]
    X, _ = quiet(sdm.sample_outcomes, n=30, seed=1, show_progress=False)[0]
    assert X["#rate"].between(0.2, 0.3).all()


# ----------------------------------------------------------------------------- I-11
def _magnitude_model(tmp_path, tag, N):
    wb = write_workbook(tmp_path / f"i11_{tag}_{N}.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, tag],
                         "Description": ["VOI", ""], "Equation": ["5 * K", NAN]},
                        [("K", "+", "S")])
    return load_model(wb, N=N, t_end=10.0)[1]


def _notices(out, warns):
    # (unrelated ResourceWarnings mention temporary file paths that contain the test name)
    text = (out + " " + " ".join(w for w in warns if not w.startswith("unclosed"))).lower()
    return text.count("larger than") + text.count("magnitude")


def test_I11_magnitude_notice_detects_large_negative_values(tmp_path):
    sdm = _magnitude_model(tmp_path, -1, 1)                  # S reaches -50
    _, out, warns = quiet(sdm.run_simulations)
    assert _notices(out, warns) >= 1


def test_I11_magnitude_notice_emitted_at_most_once_per_call(tmp_path):
    sdm = _magnitude_model(tmp_path, 1, 3)                   # S reaches +50 in each of 3 runs
    _, out, warns = quiet(sdm.run_simulations)
    assert _notices(out, warns) == 1


def test_I11_magnitude_notice_threshold_is_configurable(tmp_path):
    sdm = _magnitude_model(tmp_path, 1, 2)
    sdm.large_value_threshold = None
    _, out, warns = quiet(sdm.run_simulations)
    assert _notices(out, warns) == 0
    sdm.large_value_threshold = 100.0
    _, out, warns = quiet(sdm.run_simulations)
    assert _notices(out, warns) == 0


# ----------------------------------------------------------------------------- I-12
def test_I12_get_top_interventions_callable_on_instance(tmp_path):
    wb = write_workbook(tmp_path / "i12.xlsx",
                        {"Label": ["Y", "K"], "Type": ["auxiliary", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": [NAN, NAN]},
                        [("K", "+", "Y")])
    s, sdm = load_model(wb)
    assert sdm.get_top_interventions({"a": [1, 2], "b": [3, 4]}) == ["b", "a"]
    assert SDM.get_top_interventions({"a": [1, 2], "b": [3, 4]}, top_plot=1) == ["b"]


# ----------------------------------------------------------------------------- I-13
def test_I13_sigmoid_evaluates():
    r = EquationParser(["A", "Y"]).parse_equation("Y", "sigmoid(#1 * A)", ["A"])
    val = EquationEvaluator({"Y": r}).evaluate("Y", {"A": 1.0}, {"#1": 1.0}, 0.0)
    assert val == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-6)


# ----------------------------------------------------------------------------- I-14
@pytest.mark.slow
def test_I14_discontinuous_model_terminates_with_default_settings(tmp_path):
    """A bang-bang flow must not hang the default solver: finish or fail clearly within 60 s."""
    code = textwrap.dedent(f"""
        import io, contextlib, numpy as np, pandas as pd
        from sip_systemsinsightpipeline import Extract, SDM, IntegrationError
        p = r"{tmp_path / 'i14.xlsx'}"
        with pd.ExcelWriter(p) as xw:
            pd.DataFrame({{"Label": ["S", "F", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                          "Description": ["VOI", "", ""],
                          "Equation": [np.nan, "(S < 0.5) * 1.0 - (S >= 0.5) * 1.0 + 0 * K", np.nan]}}
                        ).to_excel(xw, sheet_name="Elements", index=False)
            pd.DataFrame([("F", "+", "S"), ("S", "-", "F"), ("K", "+", "F")], columns=["From", "Type", "To"]
                        ).to_excel(xw, sheet_name="Connections", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            s = Extract(p).extract_settings()
        s.seed, s.N, s.t_end = 1, 1, 5.0
        s.parameter_value_aux = s.parameter_value_stocks = 1.0
        sdm = SDM(s)
        params = {{"S": {{"Intercept": 0, "F": 1.0}}, "F": {{"Intercept": 0, "S": 0.0, "K": 0.0, "__eq_params_F__": {{}}}}}}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sdm.run_SDM_with_intervention_intensities([0.0], params)
            print("COMPLETED")
        except IntegrationError as e:
            print("CLEAR ERROR:", str(e)[:60])
    """)
    try:
        r = subprocess.run([sys.executable, "-c", code], timeout=60, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pytest.fail("default solver did not terminate within 60 s on a discontinuous (bang-bang) model")
    assert "COMPLETED" in r.stdout or "CLEAR ERROR" in r.stdout, r.stderr[-800:]


def test_I14_fixed_step_solver_handles_discontinuous_model(tmp_path):
    wb = write_workbook(tmp_path / "i14b.xlsx",
                        {"Label": ["S", "F", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                         "Description": ["VOI", "", ""],
                         "Equation": [NAN, "(S < 0.5) * 1.0 - (S >= 0.5) * 1.0 + 0 * K", NAN]},
                        [("F", "+", "S"), ("S", "-", "F"), ("K", "+", "F")])
    s, sdm = load_model(wb, t_end=5.0)
    sdm.solver, sdm.solver_options["dt"] = "RK4", 0.01
    params = fixed_params(sdm)
    df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, [0.0], params)
    assert abs(df["S"].iloc[-1] - 0.5) < 0.02          # chatters around the switching point


# ----------------------------------------------------------------------------- I-15
def test_I15_links_into_constants_are_ignored_consistently(tmp_path):
    wb = write_workbook(tmp_path / "i15.xlsx",
                        {"Label": ["S", "C"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""]},
                        [("C", "+", "S"), ("S", "+", "C")])
    extract = Extract(wb)
    _, _, warns = quiet(extract.extract_settings)
    assert any("constants are exogenous" in w for w in warns), warns
    assert np.array_equal(extract.adjacency_matrix, extract.df_adj.to_numpy())
    assert extract.df_adj.loc["C", "S"] == 0 and extract.df_adj.loc["S", "C"] == 1


# ----------------------------------------------------------------------------- I-16
def test_I16_equations_are_compiled_once_and_evaluator_pickles():
    r = EquationParser(["A", "Y"]).parse_equation("Y", "#1 * np.exp(A)", ["A"])
    ev = EquationEvaluator({"Y": r})
    first = ev.evaluate("Y", {"A": 0.0}, {"#1": 2.0})
    code = ev._code["Y"]
    ev.evaluate("Y", {"A": 1.0}, {"#1": 2.0})
    assert ev._code["Y"] is code and first == pytest.approx(2.0)
    clone = pickle.loads(pickle.dumps(ev))
    assert clone.evaluate("Y", {"A": 0.0}, {"#1": 3.0}) == pytest.approx(3.0)


# ----------------------------------------------------------------------------- I-17
def test_I17_stock_self_reference_does_not_warn(tmp_path):
    wb = write_workbook(tmp_path / "i17.xlsx",
                        {"Label": ["S", "K"], "Type": ["stock", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["(1 - S) * 0.5 + K", NAN]},
                        [("K", "+", "S")])
    s = extract_settings(wb)
    assert not any("{'S'}" in w for w in s.equation_warnings), s.equation_warnings


def test_I17_auxiliary_self_reference_is_rejected(tmp_path):
    wb = write_workbook(tmp_path / "i17b.xlsx",
                        {"Label": ["A", "K"], "Type": ["auxiliary", "constant"], "Tags": [0, 1],
                         "Description": ["VOI", ""], "Equation": ["0.5 * A + K", NAN]},
                        [("K", "+", "A")])
    with pytest.raises(ValueError, match="algebraic loop"):
        quiet(Extract(wb).extract_settings)


# ----------------------------------------------------------------------------- I-18
def test_I18_optimizer_respects_per_lever_caps(tmp_path):
    wb = write_workbook(tmp_path / "i18.xlsx",
                        {"Label": ["S", "L1", "L2"], "Type": ["stock", "constant", "constant"],
                         "Tags": [0, 1, 1], "Description": ["VOI", "", ""], "Equation": ["2 * L1 + L2", NAN, NAN]},
                        [("L1", "+", "S"), ("L2", "+", "S")])
    s, sdm = load_model(wb, t_end=1.0)
    params = fixed_params(sdm)
    opt = SDMOptimizer(sdm)
    free, _, _ = quiet(opt.optimize_intervention_intensities, params, [1.0, 1.0], "S", budget=1.0, n_starts=4, seed=0)
    assert free["best_intensities"][0] == pytest.approx(1.0, abs=1e-3)        # all budget on L1
    capped, _, _ = quiet(opt.optimize_intervention_intensities, params, [1.0, 1.0], "S", budget=1.0,
                         n_starts=4, seed=0, max_intensity=[0.3, np.inf])
    assert capped["best_intensities"][0] <= 0.3 + 1e-9                         # the cap holds exactly
    assert capped["best_intensities"][0] == pytest.approx(0.3, abs=1e-3)
    assert capped["best_intensities"][1] == pytest.approx(0.7, abs=1e-3)
    assert capped["best_effect_size"] == pytest.approx(2 * 0.3 + 0.7, abs=1e-3)


# ----------------------------------------------------------------------------- I-19
def test_I19_readme_recommends_an_isolated_environment():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "python -m venv" in readme


# ----------------------------------------------------------------------------- I-20
def test_I20_stock_with_flow_equation_integrates_the_flow_exactly(tmp_path):
    """Resolution of I-20: a stock that should integrate a flow gets the equation 'Flow'; its
    rate of change is then exactly the flow and no link coefficient is sampled for it."""
    wb = write_workbook(tmp_path / "i20.xlsx",
                        {"Label": ["S", "Flow", "K"], "Type": ["stock", "auxiliary", "constant"],
                         "Tags": [0, 0, 1], "Description": ["VOI", "", ""], "Equation": ["Flow", "3 * K", NAN]},
                        [("Flow", "+", "S"), ("K", "+", "Flow")])
    s, sdm = load_model(wb, N=3, t_end=2.0, pv=0.1)
    assert sdm._gsa_parameter_spec()[0] == []          # nothing uncertain left
    quiet(sdm.run_simulations)
    assert np.allclose(sdm.get_intervention_effects()["S"]["K"], 6.0, rtol=1e-6)


# ----------------------------------------------------------------------------- I-21
def test_I21_default_box_has_reasonable_coverage():
    """With a planted box plus 20% label noise, the recommended box keeps most cases of interest."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 1, size=(2000, 3)), columns=["a", "b", "c"])
    truth = (X["a"] < 0.5) & (X["b"] < 0.5)
    noise = rng.uniform(size=len(X)) < 0.2
    y = (truth ^ noise).astype(bool).values
    res = discover_scenarios(X, y, method="prim")
    assert res.box.coverage >= 0.5, repr(res.box)
    assert res.box.density == max(b.density for b in res.boxes if b.coverage >= 0.5)


# ----------------------------------------------------------------------------- I-22
LOOP_SCRIPT = textwrap.dedent("""
    import io, json, sys, contextlib
    from sip_systemsinsightpipeline import Extract, SDM, LoopsThatMatter
    p = sys.argv[1]
    with contextlib.redirect_stdout(io.StringIO()):
        s = Extract(p).extract_settings()
        s.seed, s.N, s.t_end = 1, 1, 1.0
        s.parameter_value_aux = s.parameter_value_stocks = 1.0
        ltm = LoopsThatMatter(SDM(s))
    print(json.dumps([[n, l] for n, l in zip(ltm.loop_names, ltm.feedback_loops)]))
""")


@pytest.mark.slow
def test_I22_loop_names_are_reproducible_across_processes(tmp_path):
    names = ["Habit", "Social norm", "Perceptions", "Intention", "Supply", "Taste", "Know how", "Consumption"]
    links = [(names[i], "+", names[(i + 1) % 8]) for i in range(8)]                  # ring
    links += [("Consumption", "+", "Habit"), ("Habit", "-", "Intention"), ("Supply", "+", "Social norm"),
              ("Taste", "+", "Perceptions"), ("Know how", "+", "Supply"), ("Intention", "-", "Taste"),
              ("Social norm", "+", "Consumption"), ("Perceptions", "-", "Know how")]
    links = list(dict.fromkeys(links))
    wb = write_workbook(tmp_path / "i22.xlsx",
                        {"Label": names + ["Lever"], "Type": ["stock"] * 8 + ["constant"], "Tags": [0] * 8 + [1],
                         "Description": ["VOI"] + [""] * 8, "Equation": [NAN] * 9},
                        links + [("Lever", "+", "Habit")])
    results = []
    for seed in ("1", "2", "3", "4"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", LOOP_SCRIPT, wb], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr[-800:]
        results.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert all(res == results[0] for res in results[1:]), "loop names/rotations differ between PYTHONHASHSEED values"
    # canonical form: every loop starts at its member that comes first in the model order
    order = {v: k for k, v in enumerate(names)}
    for _, loop in results[0]:
        assert loop[0] == min(loop, key=order.get)

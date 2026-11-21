"""Golden-value regression tests on the public tutorial models.

``tests/data/golden_tutorials.json`` holds, for each tutorial workbook, the simulated variable of
interest per intervention and parameter sample, the intervention effects, every variable of one
run, and the Loops That Matter scores of one parameter draw (keyed by a hash of each loop's
canonical member list, with its R/B name). The values of Minimal and the two Insulation models
were produced by SIP 0.5.0 (commit ad14fae) and are reproduced exactly by 0.6.0: the fixes of the
external review do not change results for these models (their zero state is an equilibrium, their
names contain no substrings of each other).

StockFlow (added after the 0.6.0 release) is the opposite case, a stock-and-flow model in custom
equations as applied models are written: stocks integrating their net flows, a transfer with a
named parameter shared by two equations and its range in the Parameters sheet, a policy that ramps
in with time ``t`` (so the baseline is not zero and effects are baseline-subtracted), and a name
that contains another name. Its values were produced by 0.6.0 (commit 8bd14a9).

Any change to these numbers must be deliberate. After such a change, regenerate the file with::

    python tests/test_tutorial_models.py --update              # every case
    python tests/test_tutorial_models.py --update StockFlow    # only the named case(s)

and explain the difference in the commit message.
"""
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "golden_tutorials.json"
TUTORIALS = HERE.parent / "tutorials"

CASES = {
    "Minimal": dict(file="Minimal.xlsx", seed=12345, N=5, t_end=10, pva=0.1, pvs=0.1),
    "Insulation_PeakEnergy": dict(file="Insulation_PeakEnergy.xlsx", seed=12345, N=4, t_end=10, pva=0.1, pvs=0.1),
    "Insulation_Decisions": dict(file="Insulation_Decisions.xlsx", seed=7, N=3, t_end=5, pva=0.2, pvs=0.2),
    "StockFlow": dict(file="StockFlow.xlsx", seed=12345, N=5, t_end=20, pva=0.2, pvs=0.2),
}


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def fingerprint(name):
    """Recompute the regression values of one tutorial model."""
    from sip_systemsinsightpipeline import Extract, SDM, LoopsThatMatter
    c = CASES[name]
    s = _quiet(Extract(str(TUTORIALS / c["file"])).extract_settings)
    s.seed, s.N, s.t_end, s.time_unit = c["seed"], c["N"], c["t_end"], "years"
    s.parameter_value_aux, s.parameter_value_stocks = c["pva"], c["pvs"]
    sdm = _quiet(SDM, s)
    df_sol, _ = _quiet(sdm.run_simulations)
    voi = s.variable_of_interest[0]
    effects = _quiet(sdm.get_intervention_effects)[voi]
    entry = {
        "settings": c,
        "voi": voi,
        "final_voi": {iv: [float(df_sol[n][i][voi].iloc[-1]) for n in range(s.N)]
                      for i, iv in enumerate(s.intervention_variables)},
        "effects": {k: [float(x) for x in v] for k, v in effects.items()},
        "all_final_values_sample0_first_intervention": {
            col: float(df_sol[0][0][col].iloc[-1]) for col in df_sol[0][0].columns if col != "Time"},
    }
    np.random.seed(c["seed"] + 1)
    params = _quiet(sdm.sample_model_parameters)
    ltm = _quiet(LoopsThatMatter, sdm)
    if ltm.feedback_loops:
        intensities = np.zeros(len(s.intervention_variables))
        intensities[0] = 1.0
        _, _, loops = _quiet(ltm.run_ltm_analysis, params, n_points=21,
                             intervention_intensities=intensities, progress_bar=False)
        summary = ltm.get_loop_summary(loops).set_index("loop_name")
        entry["ltm"] = {
            loop_key(loop): {"name": n,
                             "mean_relative": float(summary.loc[n, "mean_relative"]),
                             "mean_raw": float(summary.loc[n, "mean_raw"])}
            for loop, n in zip(ltm.feedback_loops, ltm.loop_names)}
    return entry


def loop_key(loop):
    """Short, content-based identity of a (canonically rotated) loop."""
    import hashlib
    return hashlib.sha1(" -> ".join(loop).encode("utf-8")).hexdigest()[:16]


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(CASES))
def test_tutorial_model_matches_golden_values(name, golden):
    expected, actual = golden[name], fingerprint(name)
    assert actual["voi"] == expected["voi"]
    assert list(actual["effects"]) == list(expected["effects"])          # same ranking order
    for key in ("final_voi", "effects", "all_final_values_sample0_first_intervention"):
        assert set(actual[key]) == set(expected[key]), key
        for k in expected[key]:
            np.testing.assert_allclose(actual[key][k], expected[key][k], rtol=1e-7, atol=1e-12,
                                       err_msg=f"{name}: {key}[{k}]")
    if "ltm" in expected:
        assert set(actual["ltm"]) == set(expected["ltm"])                 # same loops
        for loop, exp in expected["ltm"].items():
            act = actual["ltm"][loop]
            assert act["name"] == exp["name"], loop                       # stable loop IDs (I-22)
            assert act["mean_relative"] == pytest.approx(exp["mean_relative"], rel=1e-6, abs=1e-12)
            assert act["mean_raw"] == pytest.approx(exp["mean_raw"], rel=1e-6, abs=1e-12)


if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.exit("Run with --update [case ...] to regenerate tests/data/golden_tutorials.json")
    sys.path.insert(0, str(HERE.parent))
    # Only the named cases are recomputed (all cases if none is named); the others keep their
    # stored values, so that adding a model does not rewrite the others with platform noise.
    names = [arg for arg in sys.argv[1:] if arg != "--update"] or list(CASES)
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        sys.exit(f"Unknown case(s) {unknown}; the cases are {list(CASES)}")
    data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    data.update({name: fingerprint(name) for name in names})
    data = {name: data[name] for name in CASES if name in data}
    GOLDEN.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {GOLDEN} ({', '.join(names)})")

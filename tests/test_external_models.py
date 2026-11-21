"""Optional checks on private models that cannot be published with SIP.

Point the environment variable ``SIP_EXTRA_MODELS`` at a folder of Kumu-style workbooks (searched
recursively). Without it these tests are skipped. In PowerShell::

    $env:SIP_EXTRA_MODELS = "C:\\path\\to\\private_models"
    pytest tests/test_external_models.py -v

1. Every workbook is extracted, simulated for a few parameter samples and checked for finite
   results (or a clear ``IntegrationError``, never a hang); its feedback loops must be
   enumerated identically twice.
2. An optional ``sip_reference.json`` in that folder lists deterministic reference runs with
   expected values (computed by an independent implementation, or by a validated SIP version).
   See ``tests/README.md`` for the format.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sip_systemsinsightpipeline import IntegrationError, LoopsThatMatter
from sip_test_helpers import fixed_params, load_model, quiet

ROOT = os.environ.get("SIP_EXTRA_MODELS")
pytestmark = [pytest.mark.external,
              pytest.mark.skipif(not ROOT, reason="set SIP_EXTRA_MODELS to a folder of private models")]


def _workbooks():
    if not ROOT:
        return []
    return sorted(p for p in Path(ROOT).rglob("*.xlsx") if not p.name.startswith("~$"))


def _reference_cases():
    if not ROOT:
        return []
    path = Path(ROOT) / "sip_reference.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _rel(path):
    return str(Path(path).relative_to(ROOT)) if ROOT else str(path)


@pytest.mark.parametrize("workbook", _workbooks(), ids=_rel)
def test_workbook_extracts_and_simulates(workbook):
    s, sdm = load_model(workbook, N=2, t_end=1.0, pv=0.1)
    sdm.large_value_threshold = None
    try:
        quiet(sdm.run_simulations)
    except IntegrationError as exc:          # acceptable: a clear error instead of a hang
        assert "t =" in str(exc)
        return
    for voi, effects in quiet(sdm.get_intervention_effects)[0].items():
        for lever, values in effects.items():
            assert np.isfinite(np.asarray(values, dtype=float)).all(), (voi, lever)


@pytest.mark.parametrize("workbook", _workbooks(), ids=_rel)
def test_workbook_loops_are_enumerated_reproducibly(workbook):
    s, sdm = load_model(workbook, pv=0.1)
    first, _, _ = quiet(LoopsThatMatter, sdm)
    second, _, _ = quiet(LoopsThatMatter, sdm)
    assert first.feedback_loops == second.feedback_loops
    assert first.loop_names == second.loop_names


@pytest.mark.parametrize("case", _reference_cases(), ids=lambda c: c["name"])
def test_reference_run(case):
    t_end = float(case["t_end"])
    s, sdm = load_model(Path(ROOT) / case["workbook"], t_end=t_end, pv=1.0,
                        time_unit=case.get("time_unit", "years"))
    sdm.large_value_threshold = None
    sdm.solver = case.get("solver", "LSODA")
    sdm.solver_options.update(case.get("solver_options", {}))
    sdm.t_eval = np.linspace(0.0, t_end, int(case.get("n_points", 101)))
    sdm.t_span = [0.0, t_end]
    if "n_feedback_loops" in case or "loop_types" in case:
        ltm, _, _ = quiet(LoopsThatMatter, sdm)
        if "n_feedback_loops" in case:
            assert len(ltm.feedback_loops) == case["n_feedback_loops"]
        names = {" -> ".join(loop): name for loop, name in zip(ltm.feedback_loops, ltm.loop_names)}
        for loop, kind in case.get("loop_types", {}).items():
            assert loop in names, f"loop {loop!r} not found"
            assert names[loop][0] == kind, f"loop {loop!r} is {names[loop]}, expected type {kind}"
    params = fixed_params(sdm, stock_links=case.get("stock_link_coefficient", 1.0),
                          aux_links=case.get("aux_link_coefficient", 0.0),
                          equation_parameters=case.get("equation_parameters"))
    for scenario, levers in case["scenarios"].items():
        unknown = set(levers) - set(s.intervention_variables)
        assert not unknown, f"unknown levers {unknown}"
        intensities = [float(levers.get(v, 0.0)) for v in s.intervention_variables]
        expected = case["expected"][scenario]
        if expected == "IntegrationError":
            with pytest.raises(IntegrationError):
                quiet(sdm.run_SDM_with_intervention_intensities, intensities, params)
            continue
        df, _, _ = quiet(sdm.run_SDM_with_intervention_intensities, intensities, params)
        for var, values in expected.items():
            got = np.interp(case["times"], df["Time"].to_numpy(), df[var].to_numpy())
            np.testing.assert_allclose(got, values, rtol=0, atol=float(case.get("atol", 1e-6)),
                                       err_msg=f"{case['name']} / {scenario} / {var}")

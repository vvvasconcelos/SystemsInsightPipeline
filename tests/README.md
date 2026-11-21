# SIP test suite

Run everything from the repository root (inside the virtual environment where SIP is installed
with `pip install -e ".[dev]"`):

```sh
pytest                      # whole suite (about 2 minutes)
pytest -m "not slow"        # skip the subprocess tests
pytest tests/test_equations.py -v
```

## Layout

| File | What it covers |
|---|---|
| `sip_test_helpers.py` | Shared helpers: `write_workbook` (build a Kumu-style workbook from Python lists), `load_model` / `extract_settings` (quiet extraction with the usual settings), `quiet` (capture stdout and warnings), `fixed_params` (a deterministic parameter dict). |
| `conftest.py` | Fixtures (`tutorials_dir`, `data_dir`); non-interactive matplotlib backend. |
| `test_reported_issues.py` | One acceptance test per issue of the external review of 0.5.0 (`test_I01_...` to `test_I22_...`), encoding each issue's definition of done. See `bug_reports/external_review_2026-11.md`. |
| `test_equations.py` | Equation parser and evaluator: whole-name matching, punctuation in names, parameters (`#`, `#1`, `#name`), `$`, time `t`, functions, error messages. |
| `test_extraction.py` | Reading workbooks: types, tags, links into constants, variables without links, the Parameters sheet. |
| `test_simulation.py` | Simulation core: baseline-subtracted effects, time-dependent equations, solvers and the evaluation budget, auxiliary ordering, state hygiene. |
| `test_parameters.py` | Named parameters shared across equations and their ranges in every sampler (simulations, GSA, scenario discovery, optimizer, LTM). |
| `test_ltm.py` | Loops That Matter: canonical loop enumeration and names, custom-equation stocks, time. |
| `test_gsa.py`, `test_scenario_discovery.py`, `test_optimizer.py`, `test_trajectories.py` | Analysis modules. |
| `test_smoke_minimal.py`, `test_equation_linear_path.py` | End-to-end and historical regressions. |
| `test_tutorial_models.py` + `data/golden_tutorials.json` | Golden values of the public tutorial models Minimal, Insulation_PeakEnergy, Insulation_Decisions and StockFlow (simulations, effects, loop scores and names). |
| `test_tutorial_stockflow.py` | The stock-and-flow tutorial model `tutorials/StockFlow.xlsx`: stocks integrating their net flows, a transfer with a shared named parameter and the Parameters sheet, a policy that ramps in with `t` (non-zero baseline, baseline-subtracted effects), its loops, RK4, `run_GSA(method="delta")` and `sample_outcomes`. |
| `test_external_models.py` | Optional checks on private models (see below). Skipped unless `SIP_EXTRA_MODELS` is set. |

Markers (see `pytest.ini`): `slow` for tests that start subprocesses, `external` for the private
model checks.

## Adding a regression test for a new bug

1. Reproduce the bug with the smallest workbook you can build with `write_workbook`.
2. Put the test next to related tests in the module file (e.g. a parser bug in
   `test_equations.py`). For an externally reported issue, also add `test_<ID>_...` to
   `test_reported_issues.py` (or a new file per review) with the issue's definition of done.
3. Check that the test fails before the fix and passes after it.

## Golden values of the tutorial models

`test_tutorial_models.py` compares the tutorial models with `data/golden_tutorials.json`. If a
change is *meant* to alter those numbers, regenerate the file and explain why in the commit:

```sh
python tests/test_tutorial_models.py --update              # every case
python tests/test_tutorial_models.py --update StockFlow    # only the named case(s)
```

Name the cases you meant to change (or add): regenerating every case on another platform rewrites
the others with floating-point noise (about 1e-11 relative), which the tests tolerate but the diff
does not hide.

## Private models (`SIP_EXTRA_MODELS`)

Models that cannot be published can still be used as regression tests. For tests and examples in
the repository, use the public `tutorials/StockFlow.xlsx` instead: it has the structure of an
applied stock-and-flow model (stocks integrating their flows, shared named parameters, a policy in
time) without anyone's data. Put the private workbooks in a folder outside the repository and
point `SIP_EXTRA_MODELS` at it:

```powershell
$env:SIP_EXTRA_MODELS = "C:\path\to\private_models"
pytest tests/test_external_models.py -v
```

Every workbook in the folder (recursively) is extracted and simulated, and its loops are
enumerated twice. An optional `sip_reference.json` in the folder adds deterministic reference
runs:

```json
{
  "cases": [
    {
      "name": "my model: baseline and package",
      "workbook": "my_model.xlsx",
      "t_end": 25.0, "n_points": 101, "time_unit": "years",
      "times": [0, 5, 15, 25],
      "solver": "LSODA", "solver_options": {"dt": 0.01},
      "stock_link_coefficient": 1.0, "aux_link_coefficient": 0.0,
      "equation_parameters": {"Var | #1": 0.5, "#rate": 0.2},
      "scenarios": {"baseline": {}, "package": {"Lever A": 0.1, "Lever B": 0.2}},
      "expected": {"baseline": {"VOI name": [0.59, 0.58, 0.55, 0.54]},
                   "package": "IntegrationError"},
      "atol": 1e-6,
      "n_feedback_loops": 285,
      "loop_types": {"A -> B -> C": "R"}
    }
  ]
}
```

- The run uses `t_eval = linspace(0, t_end, n_points)`; values are interpolated at `times`.
- Every link into a stock without an equation gets `stock_link_coefficient`, every other link
  `aux_link_coefficient`; equation parameters are given with the names `run_GSA` uses.
- `scenarios` map lever names to intensities (unlisted levers are 0).
- `"IntegrationError"` instead of expected values means the run must stop with that error.
- `n_feedback_loops` and `loop_types` (loop written from its first variable in model order,
  `"R"` or `"B"`) are optional structural checks.

# Changelog

## Unreleased

### Added

- `tutorials/StockFlow.xlsx`, a public synthetic stock-and-flow model written entirely in custom
  equations, the way applied models are: four stocks that integrate their net flows, a transfer
  between two of them with the shared named parameter `#completion_rate`, a `Parameters` sheet, a
  local parameter, a policy that ramps in with `t` (so the baseline is not zero), balancing and
  reinforcing loops, three levers and names with punctuation. The notebook
  `tutorials/StockFlow.ipynb` walks through it (effects against the baseline, Loops That Matter,
  Borgonovo δ, `sample_outcomes`, fixed-step RK4).
- Its golden values in `tests/data/golden_tutorials.json`, and `tests/test_tutorial_stockflow.py`
  for the features it exercises. `python tests/test_tutorial_models.py --update <case> ...`
  regenerates only the named cases.

### Fixed

- `plot_simulated_intervention_ranking`: the default x-label takes two lines and a long title
  is wrapped, so both stay within the plot. The 0.6.0 label was wider than the plot area, and
  long outcome names ran past the figure. `top_plot` no longer counts the variable of interest,
  which is never drawn (`top_plot=15` showed 14 interventions in the Insulation tutorial). The
  tutorials' ranking plots and the two ranking figures of the docs were redrawn; the docs
  figures also had two titles printed over each other.

## 0.6.0 — 2026-11-21

Resolves the 22 issues of the external review of 0.5.0 (see
[bug_reports/external_review_2026-11.md](bug_reports/external_review_2026-11.md)) and three
compatibility problems with current numpy/pandas/SALib releases.

### Changed behaviour

- **Intervention effects are measured against a no-intervention baseline** (I-06).
  `get_intervention_effects()` returns `VOI(intervention) - VOI(baseline)` at the final time for
  each parameter sample; `run_simulations` also simulates the baseline
  (`sdm.df_sol_baseline_per_sample`). Pure causal-loop-diagram models are unaffected (their
  baseline is zero); models with their own dynamics no longer mix drift into effects.
  `relative_to=None` gives the old raw values. `run_SA` uses the same definition.
- **Unknown names in equations stop extraction** with a message naming them and the closest model
  variable (I-02, I-07). Before, a typo was accepted whenever any real variable occurred in the
  equation, and failed later inside the ODE solver.
- An auxiliary whose equation refers to itself is rejected (an algebraic loop); a stock may refer
  to its own level without a self-link and no longer triggers a warning (I-17).
- `ScenarioResult.box` is the densest box that keeps at least 50% of the cases of interest
  (`default_min_coverage`), no longer simply the densest box (I-21).
- Loops are enumerated in a canonical order (by length, then model order; each loop starts at
  its first variable in model order), so loop IDs (R1, B1, ...) are reproducible across sessions
  (I-22). IDs therefore differ from those printed by 0.5.0, which depended on the session.
- Links into constants are ignored with a warning (was: a printed notice), and
  `Extract.adjacency_matrix` now always matches `df_adj` (I-15).
- The "values larger than 10" notice became one `UserWarning` per call, uses `|x|`, and its
  threshold is `sdm.large_value_threshold` (`None` disables it) (I-11).
- A failed or stalled integration raises `IntegrationError` instead of hanging or failing with
  an unrelated shape error (I-14).
- `networkx >= 3.1` is required (SIP already used `simple_cycles(length_bound=...)`).

### Added

- Named equation parameters `#name`, shared by every equation that uses them, with ranges from an
  optional `Parameters` sheet (`Name | Low | High`; default `[0, parameter_value_aux]`) honoured by
  all samplers: `run_simulations`, the optimizer, Loops That Matter, `run_GSA`,
  `sample_outcomes` (I-09, I-10). Numbered `#1`, `#2`, `#` stay local to their equation.
- The simulation time `t` in equations (I-07).
- Variables with a custom equation need no incoming link, e.g. a clock stock `1` (I-08).
- Integration settings `sdm.solver_options` (`rtol`, `atol`, `max_step`, `dt`,
  `max_rhs_evals`) and fixed-step solvers `sdm.solver = "RK4"` / `"Euler"` (I-14).
- Optimizer: per-lever caps `max_intensity`; results also report `baseline_voi` and
  `best_effect_vs_baseline` (and per equilibrium `effect_vs_baseline`) (I-18, I-06).
- `run_GSA(..., relative_to="baseline")` and `sample_outcomes(..., relative_to="baseline")`
  analyse intervention effects instead of outcomes.
- `find_feedback_loops`, `IntegrationError` exported from the package;
  `plot_simulated_intervention_ranking(..., xlabel=...)`.

### Fixed

- Bare `#` parameters crashed the equation compiler; `#1` mixed with `#` collided (I-01).
- Variable names were matched as substrings, creating false dependencies and a false "circular
  dependency" error (I-03); names ending in punctuation, e.g. `(...)`, were not substituted (I-04).
- Loops That Matter crashed on stocks with a custom equation (I-05).
- `SDM.get_top_interventions` lacked `@staticmethod` (I-12).
- `sigmoid` was rewritten to the non-existent `np.sigmoid` (I-13).
- Equations are compiled once instead of re-parsed on every right-hand-side call: about 12×
  faster on a 62-variable model (I-16).
- Auxiliary ordering: interaction-term parents count as dependencies; chains deeper than 100
  auxiliaries no longer raise a false error; a real algebraic loop is reported with its members.
- An intervention on an auxiliary leaked from `run_simulations` into later runs (e.g. the Loops
  That Matter baseline).
- SALib >= 1.6 renamed the delta keys (`delta` -> `delta_raw`), breaking `run_GSA(method="delta")`.
- pandas 3 deprecation of `rename(inplace=True)`; `adjacency_matrix` and `df_adj` shared memory
  under pandas < 3, so their contents depended on the pandas version.
- Equations on constants are reported as ignored; `run_SA` no longer needs scipy >= 1.9; an SDM
  with equations can be pickled.

### Documentation and tests

- README: install into a virtual environment (I-19), custom-equation rules, named parameters and
  the Parameters sheet, how effects are defined, integration settings. Minimal tutorial: section
  on named parameters.
- Test suite reorganised (see `tests/README.md`): one acceptance test per reported issue, unit
  tests per module, golden values of the tutorial models, and an optional harness for private
  models (`SIP_EXTRA_MODELS`). Continuous integration on GitHub Actions (Python 3.9 and 3.12).

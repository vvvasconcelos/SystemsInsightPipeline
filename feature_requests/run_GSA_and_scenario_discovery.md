# Feature request — Global Sensitivity Analysis (`run_GSA`) and Scenario Discovery

**Status:** proposed · **Scope:** two new, independent analysis capabilities on top of the existing
`SDM` model-analysis workflow · **Audience:** SIP developers (self-contained — no external project context
needed).

## 1. Summary

SIP can already (a) simulate a model under parameter uncertainty (`SDM.run_simulations`,
`SDM.sample_model_parameters`), (b) rank **interventions/levers** by their effect
(`SDM.get_intervention_effects`, `compare_interventions_table`, `SDMOptimizer`), and (c) attribute behaviour
to **feedback loops** (`LoopsThatMatter`). Two standard pieces of the "which assumptions matter / where are
the risks" workflow are missing:

1. **`run_GSA` — variance-based global sensitivity analysis (Sobol).** *Which uncertain **parameters**
   drive the variance of an outcome?* First-order (`S1`) and total-order (`ST`) Sobol indices, with
   confidence intervals. This is different from the existing capabilities: intervention ranking ranks
   **levers**, GSA ranks **uncertain inputs**; and unlike one-factor-at-a-time (OFAT) it accounts for
   interactions and the full input ranges at once.

2. **Scenario discovery — `discover_scenarios`.** *Which combinations of uncertain inputs lead to an
   outcome of concern?* Given an ensemble of input draws and outcomes plus a definition of "cases of
   interest" (e.g. outcome below a threshold), find interpretable **regions** (boxes / rules) of the input
   space where those cases concentrate. This is the complement of GSA: GSA ranks drivers globally; scenario
   discovery **localises** the dangerous part of the input space.

Both are **model-agnostic**: they operate on `(inputs, outcome)` pairs the existing sampler already
produces. Neither breaks existing APIs.

---

## 2. Feature A — `SDM.run_GSA(...)`

### 2.1 Motivation
Rank the model's **uncertain parameters** by how much each contributes to the variance of a chosen outcome
(a variable of interest at the final time, or a user-supplied reducer of the trajectory). Report **S1**
(main effect) and **ST** (total effect, including interactions) with uncertainty.

### 2.2 Proposed signature
```python
def run_GSA(
    self,
    variable_of_interest=None,      # str or list[str]; defaults to self.variable_of_interest
    *,
    bounds=None,                    # dict {param_name: (low, high)}; default = the model's existing
                                    #   uniform-prior ranges (same ones sample_model_parameters uses:
                                    #   parameter_value_stocks / parameter_value_aux and equation-param bounds)
    n=1024,                         # base sample size; total model runs = n * (d + 2) for Sobol (Saltelli)
    second_order=False,            # also estimate S2 (pairwise) — costs n*(2d+2) runs
    outcome="final",               # "final" = VOI at t_eval[-1]; or "mean"/"min"/"max"; or pass reducer=
    reducer=None,                   # optional callable(df_sol_single_run) -> float, overrides `outcome`
    n_bootstrap=200,                # bootstrap resamples for the confidence intervals
    seed=None,
    show_progress=True,
):
    ...
```

### 2.3 Inputs / behaviour
- **Parameter space.** By default, use the same per-parameter **uniform bounds** the model already defines
  for sampling (`sample_model_parameters` draws `U(0, parameter_value_*)` for link coefficients and uses
  `EquationEvaluator.sample_equation_parameters` for custom-equation params). `bounds` overrides/extends
  this for named parameters. Parameters with `low == high` are treated as fixed and **excluded** from the
  indices (and reported as such).
- **Design.** Generate a **Saltelli/Sobol** sample over the `d` free parameters → evaluate the model once
  per design point (reusing the existing `run_SDM`/`run_simulations` machinery, including the custom-equation
  path) → reduce each run to a scalar outcome.
- **Estimator.** Compute `S1`, `ST` (and `S2` if requested) with the standard Sobol estimators; bootstrap
  for confidence intervals.
- **Outcome.** Default = VOI value at the last time point (matches `get_intervention_effects`). `reducer`
  lets callers target e.g. "minimum stock over the horizon" or "time-to-threshold".

### 2.4 Output
A tidy result per VOI — suggest a `pandas.DataFrame` (consistent with `compare_interventions_table`):

| parameter | S1 | S1_conf | ST | ST_conf |
|---|---|---|---|---|

…sorted by `ST` descending, plus the raw design/outputs cached on the instance (e.g. `self._gsa_design`,
`self._gsa_outputs`) so scenario discovery (Feature B) and plotting can reuse them. For multiple VOIs,
return `{voi: DataFrame}` (mirrors `get_intervention_effects`'s per-VOI dict).

### 2.5 Plotting helper (in `plots.py`)
`plot_gsa(gsa_df, kind="tornado")` — a horizontal bar chart of `ST` (with `S1` overlaid and CI whiskers),
sorted, i.e. a **Sobol tornado**. Keep it matplotlib-only and **do not call `seaborn.set_theme()`** (it
overrides a caller's style — same care as the existing plotters).

### 2.6 Implementation options (please pick one and note the dependency)
- **Option 1 — add `SALib`** (`SALib.sample.sobol.sample` + `SALib.analyze.sobol.analyze`). Mature,
  gives S1/ST/S2 + CIs out of the box. New dependency.
- **Option 2 — `scipy.stats.sobol_indices`** (needs **scipy ≥ 1.9**; SIP currently pins `scipy>=1.7`, so
  bump). No new third-party dep, but fewer conveniences (bootstrap CIs must be added).

Recommendation: **SALib** for the richer output, unless avoiding the dependency is preferred.

### 2.7 Acceptance criteria
- On the **Ishigami** test function (analytic Sobol indices known), recovered `S1`/`ST` match the reference
  within Monte-Carlo tolerance at `n=4096`.
- Indices are reproducible for a fixed `seed`; `0 ≤ S1 ≤ ST ≤ 1` (up to estimator noise); `ΣS1 ≤ 1`.
- Works for a model **with stocks** and for a model **with custom `Equation`-column parameters**.
- Fixed parameters (`low==high`) are excluded and reported; zero-variance outcome raises a clear error.
- Unit test: `tests/test_gsa.py`.

---

## 3. Feature B — `discover_scenarios(...)`

### 3.1 Motivation
Given an ensemble of uncertain-input draws `X` and a scalar outcome `y`, plus a definition of which cases
are "of interest" (e.g. `y < collapse_threshold`), find **interpretable regions** of the input space where
those cases concentrate — answering *"under what combination of conditions does it go wrong?"*. This
replaces reliance on the EMA Workbench's `prim`/`cart` so SIP is self-contained.

### 3.2 Producing the ensemble
Add a small helper so callers don't have to wire up sampling by hand (reusing existing machinery):
```python
def sample_outcomes(self, n=1000, *, bounds=None, outcome="final", reducer=None, seed=None):
    """Return (X, y): X = DataFrame [n x d] of sampled free parameters; y = outcome array [n]."""
```
(`SDMOptimizer.optimize_across_parameter_samples` already exposes its draws via
`get_last_parameter_samples()` — reuse the same sampling code so `X` columns are named consistently.)

### 3.3 Proposed signature
```python
def discover_scenarios(
    X, y,                           # X: DataFrame [n x d]; y: array [n] (continuous) or bool mask
    *,
    threshold=None,                 # if y is continuous: interesting := y < threshold (or use `predicate`)
    direction="below",             # "below" | "above"
    predicate=None,                 # optional callable(y) -> bool mask, overrides threshold/direction
    method="prim",                 # "prim" | "cart"
    peel_alpha=0.05,                # PRIM peeling fraction
    min_coverage=0.0,               # stop peeling once coverage drops below this
    max_depth=4,                    # CART only
    seed=None,
):
    ...
```

### 3.4 Output
- **PRIM:** a list of **boxes**; each box = per-dimension `(low, high)` limits actually restricted, plus
  **coverage** (recall — fraction of all interesting cases inside the box), **density** (precision —
  fraction of points in the box that are interesting), `mass` (fraction of all points in the box), and the
  **peeling trajectory** (coverage vs density at each step) so the caller can choose a box on the
  coverage–density trade-off.
- **CART:** the fitted tree plus the extracted **rules** (path → leaf) for high-density leaves, with the
  same coverage/density/mass per rule.

Return a small result object / dict; keep boxes as `DataFrame`s for readability.

### 3.5 Plotting helper (in `plots.py`)
- `plot_scenario_tradeoff(result)` — the PRIM coverage–density (peeling) curve.
- `plot_scenario_box(result, x_dim, y_dim, X, mask)` — scatter of the two chosen inputs coloured by
  interesting/not, with the selected box drawn. matplotlib-only.

### 3.6 Implementation options (note the dependency)
- **CART** → `scikit-learn` `DecisionTreeClassifier` (new optional dependency; rule extraction from the
  tree).
- **PRIM** → either vendor a small, dependency-free PRIM implementation (peeling/pasting on a hyper-box) or
  add a light dependency. A self-contained PRIM (a few hundred lines) is preferred so SIP stays installable
  without EMA Workbench.

### 3.7 Acceptance criteria
- On a **synthetic dataset** with a known interesting region (e.g. `d=5` uniform inputs,
  `interesting := x1 > 0.7 and x2 < 0.3`), PRIM returns a box whose limits on `x1`/`x2` are close to the
  truth with **high density** and meaningful **coverage**, leaving `x3..x5` unrestricted; CART recovers the
  same two splitting variables.
- Reproducible for a fixed `seed`; handles continuous inputs (categorical optional/clearly documented).
- Degenerate cases (no interesting cases / all interesting) raise clear errors.
- Unit test: `tests/test_scenario_discovery.py`.

---

## 4. Dependencies & compatibility
- New deps to declare in `setup.py`/`requirements.txt` (choose per the options above): **`SALib`** (GSA) and
  **`scikit-learn`** (scenario-discovery CART); optionally bump **`scipy>=1.9`** if using
  `scipy.stats.sobol_indices` instead of SALib. PRIM preferably vendored (no dependency).
- **No breaking changes:** both are new methods/functions; existing behaviour is untouched. Mirror existing
  conventions — `variable_of_interest` may be a list, return tidy `DataFrame`s, and **never** call
  `seaborn.set_theme()` inside plotters.
- Suggested docs: a short `docs/tutorial-gsa-and-scenario-discovery.html` mirroring the existing tutorials.

## 5. Why these two, together
GSA tells you **which assumptions your conclusion is most sensitive to** (where to spend measurement
effort); scenario discovery tells you **which combinations of conditions produce a bad outcome** (where the
risk lives). With the existing intervention ranking + Loops-That-Matter, that rounds out SIP's
"analyse → find leverage → stress-test robustness" workflow.

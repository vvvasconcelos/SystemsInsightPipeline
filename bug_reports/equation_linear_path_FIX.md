# Fix instructions: honour the `Equation` column on models without interaction terms

> **Status: APPLIED.** Implemented in `sip_systemsinsightpipeline/sdm.py` (predicate
> `self._uses_equations`, used in `run_simulations` and `run_SDM`), with the same predicate
> mirrored in `optimizer.py` and `ltm.py`. Guarded by `tests/test_equation_linear_path.py`.

Companion to `equation_linear_path.md` (diagnosis) and `equation_linear_path_repro.py` (repro).
Target file: `sip_systemsinsightpipeline/sdm.py`.

## Goal

A model whose relationships come from the Elements **`Equation`** column must simulate correctly even
when it has **no interaction terms**. Today it silently returns all-zero dynamics, because custom
equations are only evaluated on the nonlinear ODE path (`dz_dt_int`), which is taken only when
`self.interaction_terms` is non-empty.

## Chosen fix (small, safe): route equation models through the nonlinear path

`SDM._compute_equation_value` (the function `dz_dt_int` calls) is already the *single source of truth*
for evaluating a node: it uses the custom equation when one exists, and **falls back to the default
linear combination of link coefficients otherwise**. So sending an equation model — even a *mixed* one,
where some nodes use equations and some use plain links — through `dz_dt_int` is safe and correct.
`dz_dt_int` does not depend on interaction terms in any way.

So: **take the nonlinear path whenever the model has interaction terms OR any custom equation.**

### Step 1 — a "uses equations" predicate

`EquationEvaluator` already exposes `has_custom_equation(var)` and the `equations` dict. Add a tiny
helper (either on the evaluator or inline in `SDM.__init__`):

```python
# in SDM.__init__, after self._equation_evaluator is set:
self._uses_equations = bool(self._equation_evaluator) and any(
    info.get("is_valid") for info in self._equation_evaluator.equations.values()
)
```

### Step 2 — `run_simulations`: don't take the linear A/b shortcut for equation models

```python
# run_simulations(), where A, b are prepared (~L166):
-        if self.interaction_terms:
+        if self.interaction_terms or self._uses_equations:
             A = None
             b = None
         else:  # Get the system matrices A and b
             A, b = self.params_to_A_b(params, constants_values)
```

### Step 3 — `run_SDM`: integrate via `dz_dt_int` for equation models

```python
# run_SDM(), the path branch (~L193):
-        if self.interaction_terms:
+        if self.interaction_terms or self._uses_equations:
             solution = solve_ivp(self.dz_dt_int, self.t_span, x0,
                                  args=(constants_values, params),
                                  t_eval=self.t_eval, method=self.solver, rtol=1e-6, atol=1e-6).y.T
         else:  # Linear system
             ...
```

That is the whole fix. `dz_dt_int` ignores `A`/`b` (they are `None` here), evaluates each stock
derivative through `_compute_equation_value`, and the equation is applied.

> Note: the **PATCH** in `run_simulations` (~L115–120) that strips link parameters for custom-equation
> variables can stay — `_compute_equation_value` reads the equation, not those link params. (You may
> optionally keep the link params too; they are simply unused for equation nodes.)

## Verify the fix

1. **Repro flips:** `python bug_reports/equation_linear_path_repro.py` — case **A** should now print
   `S(end) ≈ 20` (matching case C), and the intervention effect should be ≈ 20.
2. **Regression test:** `pytest tests/test_equation_linear_path.py`
   - `test_equation_column_drives_linear_dynamics` is marked `xfail(strict=True)`. After the fix it will
     **XPASS**, which strict mode reports as a failure. That is the signal: **remove the
     `@pytest.mark.xfail(...)` decorator** so it becomes an ordinary passing regression test.
   - The other two tests should remain green.
3. **No regressions:** `pytest tests/` — confirm `test_smoke_minimal`, `test_extraction`, and
   `test_optimizer` still pass (interaction-term and plain-link models are unaffected by the predicate).

## Things to check while you are in there

- **Optimizer / sensitivity** (`optimize_intervention_intensities`, `optimize_across_parameter_samples`,
  `run_SA`): they go through `run_simulations` / `run_SDM`, so they inherit the fix — but confirm none of
  them assume `A`/`b` are non-`None` for an equation model.
- **Performance:** the nonlinear path uses `solve_ivp` per sample instead of the closed-form
  `analytical_solution`. That is expected and fine for typical model sizes; only flag if N is very large.
- **Fail loudly, just in case:** consider raising a warning if `self._uses_equations` is true but the
  linear branch is ever reached, so this can never regress silently again.

## Alternative (not recommended)

Make `params_to_A_b` equation-aware by linearising each custom equation into `A`/`b` (partials w.r.t.
each source for `A`, constant term for `b`). More code, and it discards any nonlinearity in the equation,
so the routing fix above is preferred.

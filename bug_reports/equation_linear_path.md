# Bug: custom `Equation` column is silently ignored on the linear SDM path

> **Status: RESOLVED.** Fixed by routing models with any valid custom equation through the
> nonlinear ODE path (see `equation_linear_path_FIX.md`). Guarded by
> `tests/test_equation_linear_path.py`. This document is retained as the diagnosis of record.

**Package:** `sip_systemsinsightpipeline` 0.2.1 (affected)
**Severity:** high (wrong results, no error/warning)
**Repro:** `python bug_reports/equation_linear_path_repro.py`

## Summary

A model that defines node relationships in the Elements **`Equation`** column but contains **no
interaction terms** runs with **all-zero dynamics** for every custom-equation variable. The equations
are parsed and validated correctly, and the stock-mediated feedback loop is detected correctly — but the
simulation ignores the equations entirely and returns zero trajectories and zero intervention effects.
**No warning or error is raised**, so the result looks plausible.

Custom equations are only evaluated on the **nonlinear** ODE path (`SDM.dz_dt_int` →
`_compute_equation_value`), which is taken only when `self.interaction_terms` is non-empty. Any purely
linear custom-equation model (the common case — e.g. a stock-and-flow model whose coefficients were
fitted from data) takes the **linear** path instead, where the equations have no effect.

## Minimal model

One stock `S` driven by one constant lever, intended dynamics `dS/dt = 2.0 * Lever`, expressed via the
`Equation` column (`"2.0 * Lever"`). With `t_end = 10` and a `Lever` intervention of 1, the correct
end-state is `2.0 × 1 × 10 = 20`.

| Case | how the link is expressed | path taken | `S(end)` | intervention effect |
|---|---|---|---|---|
| **A** | `Equation` column `"2.0 * Lever"` | linear (no interactions) | **0.000** | **0.000**  ← bug |
| **B** | ordinary signed link (no `Equation`) | linear | 1.646¹ | 1.646 |
| **C** | `Equation` column, forced nonlinear path² | nonlinear | 20.000 | 20.000 |

¹ Case B uses a *sampled* link coefficient (≈0.16), not the literal 2.0, so its magnitude differs — the
point is only that it is non-zero. ² Case C sets `sdm.interaction_terms = [(...)]` purely to route the
identical model through `dz_dt_int`; the equation is then honoured exactly (→ 20). This proves the
equation itself is correct and the linear path is the gap.

## Root cause

1. **`SDM.run_simulations` strips link parameters for custom-equation variables** (sdm.py ~L115–120):
   ```python
   if self._equation_evaluator:
       for var in params_i:
           if self._equation_evaluator.has_custom_equation(var):
               eq_keys = [k for k in params_i[var] if k.startswith('__eq_params_') or k == 'Intercept']
               params_i[var] = {k: params_i[var][k] for k in eq_keys}
   ```
   After this, `params['S'] == {'Intercept': 0, '__eq_params_S__': {}}` — the `Lever` coefficient is gone.

2. **`SDM.run_SDM` only evaluates equations on the nonlinear branch** (sdm.py ~L193–202):
   ```python
   if self.interaction_terms:
       solution = solve_ivp(self.dz_dt_int, ...)      # dz_dt_int -> _compute_equation_value (uses equations)
   else:                                              # LINEAR PATH (no interactions)
       if self.solve_analytically:
           solution = self.analytical_solution(...)   # uses A, b only
       else:
           solution = solve_ivp(self.solve_sdm_linear, ..., args=(A, b))   # uses A, b only
   ```

3. **`SDM.params_to_A_b` has no equation awareness** (sdm.py ~L307–397): it fills the system matrix
   purely from per-link coefficients keyed by source-variable names
   (`param_mat[target_row, source_col] = val`). The leftover `__eq_params_*` keys are not stock/aux/const
   names, so they are skipped, and the custom-equation variable's rows in `A` and `b` stay zero.

The repro prints the resulting system for case A directly:
```
params['S'] after the strip : {'Intercept': 0, '__eq_params_S__': {}}
A = [[0.0]]   b = [0.0]    ->  dS/dt = 0  for all t
```

So: equations are stripped from the link params (step 1), never re-injected into `A`/`b` (step 3), and
only the nonlinear path would have used them (step 2) — which a linear model never reaches.

## Suggested fixes (either)

- **Preferred / simplest:** route any model that has custom equations through the nonlinear ODE path,
  regardless of `interaction_terms` — i.e. in `run_SDM`, take the `dz_dt_int` branch when
  `self._equation_evaluator` has any custom equation. (`run_simulations` would also need to skip the
  `A, b = params_to_A_b(...)` shortcut in that case.)
- **Alternative:** make the linear path equation-aware — linearise each custom equation into `A`/`b`
  (partial derivatives w.r.t. each source variable for `A`, constant term for `b`). More work, and loses
  any nonlinearity in the equation, so the first option is cleaner.
- **In any case:** raise a warning/error if custom equations are present but the linear path is taken, so
  the failure is never silent.

## Notes

- Independent of literals vs sampled `#` parameters: a literal equation (`"2.0 * Lever"`) stores no
  sampled params (`__eq_params_S__ == {}`), but even with `#` params the linear path would not read them.
- Parsing, validation, and loop detection all work — only the *run* is affected.

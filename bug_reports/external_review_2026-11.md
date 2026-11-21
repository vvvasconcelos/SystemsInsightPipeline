# External review of SIP 0.5.0 (November 2026): verification and resolution

An external user team encoded two applied system-dynamics models in SIP custom equations, ran
them through every analysis, cross-checked the results against independent implementations, and
reported 22 issues against commit `ad14fae` (0.5.0), each with a reproduction, a suggested fix and
a definition of done.

**Verification.** Every issue was reproduced on `ad14fae` before any change: the reviewers'
probes P1–P15 gave the reported output, their 17 regression tests failed for the stated reasons,
and SIP's own suite gave 45 passed / 1 failed (I-15) with numpy 1.26 / pandas 2.2 / SALib 1.5.
The private applied models were used as additional (local, unpublished) regression tests; with
0.6.0 they reproduce the reviewers' independent reference values (to 5×10⁻⁹ for the larger model,
fixed-step RK4 to 10 decimals).

**Resolution.** All 22 are resolved in 0.6.0. Issues that needed a design decision were decided
by the maintainer (column *Decision*). Each issue has an acceptance test `test_I<nn>_...` in
`tests/test_reported_issues.py`.

| ID | Sev. | Issue | Resolution | Decision |
|---|---|---|---|---|
| I-01 | High | Bare `#` parameters crash; `#1` + `#` collide | Bare `#` numbered once, after the highest explicit number | |
| I-02 | Med. | Typos not detected | Expressions validated with Python's `ast`; unknown names stop extraction with "did you mean" | |
| I-03 | High | Substring matching of names → false circular dependency | Left-to-right, longest-first whole-name scanner; dependency sort uses it | |
| I-04 | Med. | Names with punctuation not substituted | Supported (names are replaced before parsing) | support arbitrary names (except `+`, `*`) |
| I-05 | High | LTM crashes on custom-equation stocks | LTM evaluates every variable with an equation through the evaluator | |
| I-06 | High | Effects not baseline-subtracted | Baseline per parameter sample; effects = intervention − baseline by default | baseline-subtracted default; `relative_to=None` for raw values |
| I-07 | Med. | No time variable; `t` passed validation | `t` available in equations, LTM and post-processing | expose `t` |
| I-08 | Low | Self-contained equations need a link | Variables with an equation may have no incoming link | allow |
| I-09 | Med. | `#` parameters scoped per equation | Named parameters `#name` shared across equations | `#name` global; `#1`, `#` stay local |
| I-10 | Med. | Priors only in GSA | Optional `Parameters` sheet (`Name, Low, High`), honoured by all samplers | default `[0, parameter_value_aux]`; numbered parameters keep the default |
| I-11 | Low | ">10" notice: no negatives, per run | One warning per call, `|x|`, `sdm.large_value_threshold` | |
| I-12 | Low | `get_top_interventions` signature | `@staticmethod` | |
| I-13 | Med. | `sigmoid` → `np.sigmoid` | SIP's own `sigmoid` (scipy `expit`) | |
| I-14 | Med. | Hard-coded solver can hang | `solver_options`, fixed-step `RK4`/`Euler`, `max_rhs_evals` budget → `IntegrationError` | LSODA stays default; budget 200,000 evaluations |
| I-15 | Low | Test expected links into constants | Links into constants ignored with a warning; test fixed; matrices kept consistent | keep removing links into constants |
| I-16 | Low | `eval` re-parses equations per call | Code objects compiled once (1.15 → 0.10 ms per RHS call on a 62-variable applied model) | |
| I-17 | Low | Stock self-reference warns | No warning for stocks; auxiliary self-reference rejected | implicit self-reference for stocks |
| I-18 | Low | Optimiser has no per-lever caps | `max_intensity` (scalar or per lever) | caps only; no nonlinear costs |
| I-19 | Low | Recommend a virtual environment | README install section | |
| I-20 | Med. | Known link coefficients cannot be fixed | Documented: give the stock the equation `Flow` (no sampled coefficient); otherwise sampling applies | no new workbook column |
| I-21 | Low | Default PRIM box ignores coverage | Densest box with coverage ≥ 0.5 (`default_min_coverage`) | |
| I-22 | Med. | Loop IDs not reproducible | Canonical enumeration (integer graph, rotation, sort) | |

## Further findings during verification

| Finding | Resolution |
|---|---|
| SALib ≥ 1.6 returns `delta_raw` instead of `delta`; `run_GSA(method="delta")` raised `KeyError` | Both key sets supported |
| Under pandas < 3, `Extract.df_adj` shared memory with `adjacency_matrix`, so I-15's test passed or failed depending on the pandas version | Explicit copies; both always agree |
| pandas 3 deprecates `DataFrame.rename(inplace=True)` | Replaced |
| `run_simulations` left the last auxiliary intervention active, so a later Loops That Matter baseline run silently included it | Intervention state restored after every run |
| `sort_auxiliaries` ignored interaction-term parents and gave up after 100 dependency levels | Fixed; real cycles are reported with their members |
| Equations on constants were silently ignored | Warning |
| `simple_cycles(length_bound=...)` needs networkx ≥ 3.1 but `setup.py` allowed 2.5 | Requirement raised |

# Systems Insight Pipeline (SIP)

**From a causal loop diagram in Excel to quantitative, decision-ready insight.**

SIP converts causal loop diagrams (CLDs) — for example participatory diagrams drawn in
[Kumu](https://kumu.io) — into computational system dynamics models, and then tells you which
interventions matter: it simulates the system under parameter uncertainty, ranks interventions,
optimally allocates a budget across them, and identifies the feedback loops that dominate the
dynamics. No equations are required to get started — tweaking the Excel file *is* the modeling
interface — but custom equations are supported when you need them.

## What it can do

- **CLD → system dynamics model.** Reads a Kumu-style Excel export (variables, links,
  polarities) and builds a stock-auxiliary-constant ODE model automatically
  ([`Extract`](sip_systemsinsightpipeline/cld.py), [`SDM`](sip_systemsinsightpipeline/sdm.py)).
- **Simulation under uncertainty.** Link coefficients are unknown in qualitative diagrams, so SIP
  samples them and simulates every tagged intervention across `N` parameter draws
  (`SDM.run_simulations`), with trajectory plots for any variable — median/percentile bands or
  spaghetti (`plots.plot_trajectories`; see
  [docs/trajectories-and-archetypes.html](docs/trajectories-and-archetypes.html) for the classic
  trajectory shapes and the loop structures behind them).
- **Intervention ranking.** Distributions of each intervention's effect on your variable of
  interest (the change relative to a no-intervention run with the same parameters), with box
  plots, pairwise comparison tables (bootstrapped CIs, Cliff's delta), and Spearman sensitivity
  analysis of which links drive the outcome (`get_intervention_effects`,
  `compare_interventions_table`, `run_SA`).
- **Budget optimization.** Rather than one intervention at a time, allocate a budget across all
  interventions at once ([`SDMOptimizer`](sip_systemsinsightpipeline/optimizer.py)). The
  optimizer works in expenditure space (`y_i = cost_i × intensity_i`, `y ≥ 0`,
  `Σy ≤ budget`) with Sobol-seeded multi-start SLSQP, reports near-optimal alternative
  allocations, and scales to dozens of interventions. Maximize or minimize the outcome.
- **Loops That Matter.** Scores every feedback loop's contribution to model behavior over time
  (Schoenberg–Davidsen–Eberlein method; [`LoopsThatMatter`](sip_systemsinsightpipeline/ltm.py)).
- **Global sensitivity analysis.** Which uncertain parameters drive the outcome, via
  `SDM.run_GSA(method=...)`: variance-based **Sobol** indices (`S1`/`ST` with BCa bootstrap
  confidence intervals projected onto `[0,1]`), and moment-independent **Borgonovo δ** / **PAWN**
  measures that capture shifts in the whole output distribution and are non-negative by
  construction. See [docs/sensitivity-methods.html](docs/sensitivity-methods.html) for choosing
  between them.
- **Scenario discovery.** PRIM (vendored) and CART localise the combinations of conditions that
  produce an outcome of concern as interpretable boxes/rules with density and coverage
  (`discover_scenarios`, `SDM.sample_outcomes`).
- **Custom equations.** Override the default linear link aggregation for any variable with an
  `Equation` column entry — sampled parameters (local or shared across equations, with ranges
  from an optional `Parameters` sheet), intervention placement, simulation time, numpy functions
  (see below).

## Install

Requires Python ≥ 3.9. Install SIP into its own virtual environment, so that it cannot clash
with (or be broken by) packages in an existing environment such as Anaconda's `base`:

```sh
git clone https://github.com/vvvasconcelos/SystemsInsightPipeline.git
cd SystemsInsightPipeline
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux),
then:

```sh
pip install -e .
```

With conda, `conda create -n sip python=3.12` followed by `conda activate sip` works as well.

## Quick start

```python
from sip_systemsinsightpipeline import Extract, SDM, SDMOptimizer
from sip_systemsinsightpipeline.plots import plot_simulated_intervention_ranking

# 1. Load a Kumu-style Excel model
s = Extract("tutorials/Minimal.xlsx").extract_settings()
s.seed = 12345                  # reproducibility
s.N = 100                       # parameter samples
s.t_end = 10                    # time horizon
s.time_unit = "years"
s.parameter_value_aux = 0.1     # sampling bound for auxiliary/equation parameters
s.parameter_value_stocks = 0.1  # sampling bound for stock parameters

# 2. Simulate all tagged interventions under parameter uncertainty
sdm = SDM(s)
df_sol_per_sample, param_samples = sdm.run_simulations()
effects = sdm.get_intervention_effects()  # change vs. a no-intervention run

voi = s.variable_of_interest[0]
plot_simulated_intervention_ranking(s, effects[voi], voi)

# 3. Optimally allocate a budget across interventions
params = sdm.sample_model_parameters()
result = SDMOptimizer(sdm).optimize_intervention_intensities(
    params=params,
    costs=[1.0] * len(s.intervention_variables),
    variable_of_interest=voi,
    budget=1.0,
    maximize=True,   # False to minimize the outcome
    n_starts=8,
    seed=1,
)
print(result["best_effect_size"], result["best_intensities"])
```

## Tutorials

| Notebook | What it shows |
|----------|---------------|
| [`tutorials/Minimal.ipynb`](tutorials/Minimal.ipynb) | Start here: the Excel format and the full workflow on a 3-variable model (< 1 min) |
| [`tutorials/Insulation.ipynb`](tutorials/Insulation.ipynb) | A real-world model of household peak energy use from the [SEVEN project](https://seven.uva.nl/): ranking, sensitivity analysis, budget minimization over 51 interventions, Loops That Matter |
| [`tutorials/GSA_and_Scenario_Discovery.ipynb`](tutorials/GSA_and_Scenario_Discovery.ipynb) | Global sensitivity analysis (Sobol) and scenario discovery (PRIM/CART) on the insulation model: which assumptions drive the outcome, and which conditions make it go wrong |

The Alzheimer's disease example from the D2D paper (see Citation) is available in the upstream
[jerul/systemdynamics](https://github.com/jerul/systemdynamics) repository.

## The Excel format

A model is one Excel workbook, structured like a Kumu export:

**Elements sheet** — one row per variable.

| Column | Meaning |
|--------|---------|
| `Label` | Variable name. Spaces and punctuation such as `.`, `(`, `)`, `'` and `-` are fine; `+` and `*` are not allowed |
| `Type` | `stock` (accumulates), `auxiliary` (instantaneous), or `constant` (exogenous); case-insensitive. Untyped variables default to stock if they have incoming links, constant otherwise |
| `Tags` | Nonzero ⇒ the variable is an intervention with that relative strength (sign = direction) |
| `Description` | `VOI` marks the variable of interest (the outcome) |
| `Equation` | *(optional)* custom equation, see below |

**Connections sheet** — one row per causal link: `From`, `To`, `Type` (`+` or `-` polarity).
Every stock and auxiliary needs at least one incoming link, unless it has a custom equation
(e.g. a clock stock with equation `1`). Constants are exogenous: links *into* a constant are
ignored, with a warning.

**Interactions sheet** — *(optional)* pairwise interaction terms: `From1`, `From2`, `To`, `Type`.

**Parameters sheet** — *(optional)* ranges of named equation parameters: `Name`, `Low`, `High`
(see below).

### Custom equations

Add an `Equation` column entry to override the default linear combination of a variable's
incoming links. For an auxiliary the equation gives its value; for a stock, its rate of change.

- **Variables**: names exactly as in the Elements sheet (case-sensitive). Names are recognised
  whole and longest first, so `Price` and `Price index` can coexist; runs of spaces count as one.
- **Local parameters**: `#1`, `#2`, … or a bare `#` (numbered after the highest explicit number)
  — sampled uniformly from `[0, parameter_value_aux]`. They are local: `#1` in one equation and
  `#1` in another are different parameters.
- **Named parameters**: `#rate`, `#k_decay`, … (a letter after `#`, then letters, digits or
  `_`). The same name is the *same parameter in every equation* that uses it, with one value per
  parameter draw — e.g. a rate that drains one stock and fills another. Its range comes from the
  Parameters sheet, otherwise `[0, parameter_value_aux]`; `Low = High` fixes its value. The
  ranges are honoured by `run_simulations`, the optimizer, Loops That Matter, `run_GSA` and
  `sample_outcomes` (which report it as `#rate`).
- **Intervention**: `$` marks where the intervention enters (appended to the result if absent).
- **Time**: `t` is the simulation time, e.g. `K * (t > 2)` for a step at `t = 2`.
- **Functions**: `np.exp`, `np.log`, `np.tanh`, `np.sqrt`, `np.minimum`, `np.where`, …, the same
  names without `np.`, and `sigmoid`. Comparisons (`(S < 0.5)` is 1 or 0) and
  `a if condition else b` are allowed.

```text
#1 * A + #2 * B                  # linear with sampled coefficients
np.exp(-#1 * A) * B              # exponential decay
#1 * np.tanh(#2 * (A - B))       # bounded S-curve
$ + #1 * (1 - np.exp(-#2 * A))   # saturation, intervention added explicitly
(1 - S) * #rate                  # a stock S seeking 1 (refers to itself: no self-link needed)
Inflow - Outflow                 # a stock integrating its flows exactly
```

| Parameters sheet: `Name` | `Low` | `High` |
|---|---|---|
| `rate` | 0.1 | 0.4 |
| `k_decay` | 0.05 | 0.05 |

Equations are validated when the workbook is read. Names that are not model variables (typos),
unknown functions and syntax errors stop extraction with a message that names the offending text
(and the closest model variable). SIP warns when an equation uses variables that are not
incoming links, or ignores links that exist in the diagram; a stock may refer to its own level
without a self-link, while an auxiliary that refers to itself is rejected (an algebraic loop).

To make a stock integrate a flow exactly, give the stock the equation `Flow` (or
`Inflow - Outflow`). Without an equation, every incoming link has an uncertain, sampled
coefficient — the right default for a qualitative diagram, but not for a known stock-and-flow
structure.

### How intervention effects are defined

`get_intervention_effects()` reports, for every parameter sample, the value of the variable of
interest at the final time *minus* its value in a no-intervention run with the same parameters
(`run_simulations` simulates that baseline, in `sdm.df_sol_baseline_per_sample`). In a pure
causal-loop-diagram model the zero state is an equilibrium, so the baseline stays zero and the
effect equals the final value. In models with their own dynamics (custom equations with drift,
targets or time) the subtraction removes the drift from the effect. `relative_to=None` returns the
raw final values. Ranking plots, `compare_interventions_table` and `run_SA` use the same
definition. `run_GSA` and `sample_outcomes` analyse the outcome itself by default, or the effect
with `relative_to="baseline"`. The optimizer maximises or minimises the final value and also
reports `baseline_voi` and `best_effect_vs_baseline`; `max_intensity` caps individual levers.

### Integration settings

Models with custom equations or interaction terms are integrated numerically with
`sdm.solver = "LSODA"` (default) or any other `scipy.integrate.solve_ivp` method, or with the
fixed-step methods `"RK4"` and `"Euler"` that system dynamics tools use. Options live in
`sdm.solver_options`: `rtol`, `atol`, `max_step`, `dt` (fixed-step size) and `max_rhs_evals`
(200,000 by default). Equations that switch abruptly (e.g. `(S < 0.5)`) can make adaptive
solvers take ever smaller steps; SIP then stops with an `IntegrationError` that says where it
stalled, instead of hanging. A fixed-step method handles such models predictably:

```python
sdm.solver = "RK4"
sdm.solver_options["dt"] = 0.01
```

When simulated stock values exceed `sdm.large_value_threshold` (default 10, in absolute value)
SIP issues one warning per call; set it to `None` for models in absolute units.

## Development

```sh
pip install -e ".[dev]"
pytest
```

The test suite is described in [tests/README.md](tests/README.md): one acceptance test per
reported issue, unit tests per module, golden values of the tutorial models, and an optional
harness that runs private models from a local folder (`SIP_EXTRA_MODELS`). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## Origin and citation

This repository started as a fork of
[jerul/systemdynamics](https://github.com/jerul/systemdynamics) by Jeroen F. Uleman, which
implements the Diagrams-to-Dynamics (D2D) method, and extends it with budget optimization,
Loops That Matter analysis, and custom equation support. The insulation tutorial models were
developed in the [SEVEN project](https://seven.uva.nl/) at the University of Amsterdam (see the
[project report](https://seven.uva.nl/en/content/news/2026/03/kgg-report.html)).

If you use this software in your research, please cite the D2D paper:

> Uleman, J.F., Crielaard, L., Elsenburg, L.K., Veldhuis, G.A., Rod, N.H., Quax, R., &
> Vasconcelos, V.V. *Diagrams-to-Dynamics (D2D): Exploring Causal Loop Diagram Leverage Points
> under Uncertainty.* BMC Medicine (in press).
> Preprint: [arXiv:2508.05659](https://arxiv.org/abs/2508.05659).

```bibtex
@article{uleman2026d2d,
  title   = {Diagrams-to-Dynamics (D2D): Exploring Causal Loop Diagram Leverage Points under Uncertainty},
  author  = {Uleman, Jeroen F. and Crielaard, Loes and Elsenburg, Leonie K. and Veldhuis, Guido A. and Rod, Naja Hulvej and Quax, Rick and Vasconcelos, V{\'i}tor V.},
  journal = {BMC Medicine},
  year    = {2026},
  note    = {In press. Preprint: arXiv:2508.05659},
  url     = {https://arxiv.org/abs/2508.05659}
}
```

## License

[GPL-3.0](LICENSE), inherited from the upstream repository.

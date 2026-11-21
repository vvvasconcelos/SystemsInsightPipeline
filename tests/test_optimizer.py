import numpy as np
import pytest
import scipy.optimize
from scipy.stats import qmc

from sip_systemsinsightpipeline.optimizer import SDMOptimizer
from sip_test_helpers import NAN, fixed_params, load_model, quiet, write_workbook


def test_slsqp_converges_with_large_finite_penalty():
    """Regression for the optimizer's design choice of returning a large finite
    penalty instead of np.inf: SLSQP must still converge to the true optimum."""
    def objective(x):
        if x[0] < 0.3:  # penalty region
            return 1e10
        return (x[0] - 0.5) ** 2

    result = scipy.optimize.minimize(
        objective,
        x0=[0.5],
        method="SLSQP",
        bounds=[(0, 1)],
        constraints={"type": "ineq", "fun": lambda x: 0.9 - x[0]},
        options={"maxiter": 100},
    )

    assert result.success
    assert np.isfinite(result.fun)
    assert abs(result.x[0] - 0.5) < 1e-4


def test_sobol_to_simplex_feasibility():
    """All folded Sobol starting points must satisfy the budget simplex constraint.
    (Naive hypercube sampling leaves only ~1/d! of points feasible, which is why the
    optimizer folds points into the simplex instead.)"""
    d = 4
    u = qmc.Sobol(d=d + 1, scramble=False).random(64)

    for budget in (1.0, 2.5):
        y = SDMOptimizer._sobol_to_simplex_y(u, budget=budget)
        assert y.shape == (64, d)
        assert np.all(y >= 0)
        assert np.all(y.sum(axis=1) <= budget + 1e-9)


def test_sobol_to_simplex_deterministic():
    u = qmc.Sobol(d=5, scramble=False).random(32)
    y1 = SDMOptimizer._sobol_to_simplex_y(u, budget=1.0)
    y2 = SDMOptimizer._sobol_to_simplex_y(u, budget=1.0)
    assert np.array_equal(y1, y2)


# ----------------------------------------------------------------------------- caps and baseline (I-18, I-06)
def _drift_model(tmp_path):
    """dS/dt = 1 + 2 L1 + L2: a drift of 1 per time unit plus two levers."""
    wb = write_workbook(tmp_path / "opt.xlsx",
                        {"Label": ["S", "L1", "L2"], "Type": ["stock", "constant", "constant"],
                         "Tags": [0, 1, 1], "Description": ["VOI", "", ""],
                         "Equation": ["1 + 2 * L1 + L2", NAN, NAN]},
                        [("L1", "+", "S"), ("L2", "+", "S")])
    s, sdm = load_model(wb, t_end=1.0)
    return sdm, fixed_params(sdm)


def test_optimizer_reports_baseline(tmp_path):
    sdm, params = _drift_model(tmp_path)
    res, _, _ = quiet(SDMOptimizer(sdm).optimize_intervention_intensities, params, [1.0, 1.0], "S",
                      budget=1.0, n_starts=2, seed=0)
    assert res["baseline_voi"] == pytest.approx(1.0)
    assert res["best_effect_size"] == pytest.approx(3.0, abs=1e-4)
    assert res["best_effect_vs_baseline"] == pytest.approx(2.0, abs=1e-4)
    assert res["equilibria"][0]["effect_vs_baseline"] == pytest.approx(2.0, abs=1e-4)


def test_optimizer_caps_can_be_scalar_and_are_validated(tmp_path):
    sdm, params = _drift_model(tmp_path)
    opt = SDMOptimizer(sdm)
    res, _, _ = quiet(opt.optimize_intervention_intensities, params, [1.0, 1.0], "S", budget=1.0,
                      n_starts=2, seed=0, max_intensity=0.4)
    assert np.all(res["best_intensities"] <= 0.4 + 1e-9)
    assert res["best_effect_size"] == pytest.approx(1 + 2 * 0.4 + 0.4, abs=1e-4)   # budget not exhausted
    for bad in ([0.1], [-1.0, 1.0]):
        with pytest.raises(ValueError, match="max_intensity"):
            opt.optimize_intervention_intensities(params, [1.0, 1.0], "S", max_intensity=bad)


def test_optimize_across_samples_forwards_caps_and_baseline(tmp_path):
    sdm, _ = _drift_model(tmp_path)
    df, _, _ = quiet(SDMOptimizer(sdm).optimize_across_parameter_samples, [1.0, 1.0], "S",
                     n_parameter_samples=2, n_starts=1, max_intensity=[0.2, 0.2], show_progress=False)
    assert (df["intensity_L1"] <= 0.2 + 1e-9).all() and (df["intensity_L2"] <= 0.2 + 1e-9).all()
    assert np.allclose(df["baseline_voi"], 1.0)
    assert np.allclose(df["voi_effect_vs_baseline"], df["voi_effect_size"] - 1.0)

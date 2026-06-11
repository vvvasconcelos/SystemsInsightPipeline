import numpy as np
import scipy.optimize
from scipy.stats import qmc

from sip_systemsinsightpipeline.optimizer import SDMOptimizer


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

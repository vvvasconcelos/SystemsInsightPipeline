"""Tests for scenario discovery (vendored PRIM + scikit-learn CART).

The known-box acceptance test uses a synthetic dataset where the interesting region is
exactly ``x1 > 0.7 and x2 < 0.3`` over five independent uniform inputs.
"""
import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline import SDM, Extract
from sip_systemsinsightpipeline.scenario_discovery import discover_scenarios


def synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(0, 1, size=(n, 5)), columns=[f"x{i+1}" for i in range(5)])
    interesting = (X["x1"] > 0.7) & (X["x2"] < 0.3)
    return X, interesting.values


# --------------------------------------------------------------------------- PRIM

def test_prim_recovers_known_box():
    X, y = synthetic()
    res = discover_scenarios(X, y, method="prim", peel_alpha=0.05, min_coverage=0.5)
    box = res.box

    # The true drivers x1, x2 are restricted; the irrelevant x3..x5 are left alone.
    assert set(box.restricted_dimensions) == {"x1", "x2"}
    assert box.limits.loc["x1", "min"] == pytest.approx(0.7, abs=0.06)
    assert box.limits.loc["x2", "max"] == pytest.approx(0.3, abs=0.06)
    # High precision and meaningful recall.
    assert box.density > 0.9
    assert box.coverage > 0.5
    # The peeling trajectory trades coverage for density.
    assert res.trajectory["density"].iloc[-1] >= res.trajectory["density"].iloc[0]


def test_prim_deterministic():
    X, y = synthetic()
    a = discover_scenarios(X, y, method="prim", min_coverage=0.5).box
    b = discover_scenarios(X, y, method="prim", min_coverage=0.5).box
    assert a.density == b.density and a.coverage == b.coverage
    assert a.restricted_dimensions == b.restricted_dimensions


def test_prim_continuous_threshold():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.uniform(0, 1, size=(1500, 3)), columns=["a", "b", "c"])
    yv = X["a"].values + 0.01 * rng.standard_normal(1500)  # outcome driven by 'a'
    res = discover_scenarios(X, yv, threshold=0.2, direction="below",
                             method="prim", min_coverage=0.5)
    assert "a" in res.box.restricted_dimensions
    assert res.box.density > 0.8


def test_predicate_overrides_threshold():
    X, y = synthetic()
    res = discover_scenarios(X, y, predicate=lambda v: v.astype(bool), method="prim",
                             min_coverage=0.5)
    assert set(res.box.restricted_dimensions) == {"x1", "x2"}


# --------------------------------------------------------------------------- CART

def test_cart_finds_splitting_variables():
    X, y = synthetic()
    res = discover_scenarios(X, y, method="cart", max_depth=4, min_density=0.8, seed=0)
    assert res.splitting_variables == ["x1", "x2"]
    assert res.rules
    assert res.box.density > 0.9


# --------------------------------------------------------------------------- degenerate

def test_no_interesting_cases_raises():
    X, _ = synthetic()
    with pytest.raises(ValueError, match="No cases of interest"):
        discover_scenarios(X, np.zeros(len(X), dtype=bool), method="prim")


def test_all_interesting_raises():
    X, _ = synthetic()
    with pytest.raises(ValueError, match="does not discriminate"):
        discover_scenarios(X, np.ones(len(X), dtype=bool), method="prim")


def test_missing_threshold_raises():
    X, _ = synthetic()
    with pytest.raises(ValueError, match="threshold"):
        discover_scenarios(X, np.linspace(0, 1, len(X)), method="prim")


# --------------------------------------------------------------------------- SDM integration

def test_sample_outcomes_feeds_discovery(tmp_path):
    df_e = pd.DataFrame({
        "Label": ["Outcome", "DriverA", "DriverB"],
        "Type": ["stock", "constant", "constant"],
        "Tags": [0, 1, 1],
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({"From": ["DriverA", "DriverB"], "Type": ["+", "+"],
                         "To": ["Outcome", "Outcome"]})
    path = tmp_path / "stock.xlsx"
    with pd.ExcelWriter(path) as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)

    s = Extract(str(path)).extract_settings()
    s.seed = 0
    s.N = 1
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_stocks = 0.3
    s.parameter_value_aux = 0.3
    sdm = SDM(s)

    X, y = sdm.sample_outcomes(n=400, seed=1, show_progress=False)
    assert list(X.columns) == ["Outcome <- DriverA", "Outcome <- DriverB"]
    assert len(y) == 400 and np.isfinite(y).all()

    # Low outcomes occur when both link coefficients are small -> PRIM should restrict them down.
    res = discover_scenarios(X, y, threshold=float(np.quantile(y, 0.2)),
                             direction="below", method="prim", min_coverage=0.5)
    assert res.box.density > 0.5
    assert len(res.box.restricted_dimensions) >= 1


def test_scenario_plots_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    from sip_systemsinsightpipeline.plots import plot_scenario_tradeoff, plot_scenario_box

    X, y = synthetic()
    res = discover_scenarios(X, y, method="prim", min_coverage=0.5)
    fig1 = plot_scenario_tradeoff(res)
    fig2 = plot_scenario_box(res, "x1", "x2", X)
    assert fig1 is not None and fig2 is not None

"""Tests for variance-based global sensitivity analysis (Sobol indices).

The engine is validated against the analytic Ishigami indices; the SDM integration is
checked on a stock model and a custom-Equation model.
"""
import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline import SDM, Extract, gsa


# --------------------------------------------------------------------------- engine

def test_engine_recovers_ishigami_indices():
    """On the Ishigami function the recovered S1/ST match the analytic reference."""
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    names = ["x1", "x2", "x3"]
    bounds = [(-np.pi, np.pi)] * 3
    df, design, out = gsa.run_sobol(
        names, bounds, lambda r: Ishigami.evaluate(r.reshape(1, -1))[0],
        n=4096, seed=7,
    )
    df = df.set_index("parameter")
    # Analytic Ishigami (a=7, b=0.1): S1 = [0.314, 0.442, 0.0], ST = [0.558, 0.442, 0.244]
    analytic_s1 = {"x1": 0.314, "x2": 0.442, "x3": 0.0}
    analytic_st = {"x1": 0.558, "x2": 0.442, "x3": 0.244}
    for p in names:
        assert df.loc[p, "S1"] == pytest.approx(analytic_s1[p], abs=0.06)
        assert df.loc[p, "ST"] == pytest.approx(analytic_st[p], abs=0.06)
    # x3 has ~zero first-order but non-trivial total effect (x1-x3 interaction).
    assert df.loc["x3", "ST"] - df.loc["x3", "S1"] > 0.15


def test_engine_index_bounds_and_reproducible():
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    names = ["x1", "x2", "x3"]
    bounds = [(-np.pi, np.pi)] * 3
    f = lambda r: Ishigami.evaluate(r.reshape(1, -1))[0]
    a, _, _ = gsa.run_sobol(names, bounds, f, n=1024, seed=11)
    b, _, _ = gsa.run_sobol(names, bounds, f, n=1024, seed=11)
    pd.testing.assert_frame_equal(a, b)
    assert (a["S1"] <= a["ST"] + 0.05).all()
    assert (a["ST"] <= 1.05).all()
    assert a["S1"].sum() <= 1.05


def test_engine_zero_variance_raises():
    problem = gsa.sobol_problem(["a", "b"], [(0, 1), (0, 1)])
    Y = np.full(gsa.sobol_sample(problem, 16, seed=0).shape[0], 3.0)  # constant outcome
    with pytest.raises(ValueError, match="zero variance"):
        gsa.sobol_analyze(problem, Y)


# --------------------------------------------------------------------------- SDM models

def _settings(path, seed=0, **over):
    s = Extract(str(path)).extract_settings()
    s.seed = seed
    s.N = 1
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_aux = over.get("pva", 0.3)
    s.parameter_value_stocks = over.get("pvs", 0.3)
    return s


def write_stock_model(path):
    """One stock driven by two constant levers via ordinary signed links."""
    df_e = pd.DataFrame({
        "Label": ["Outcome", "DriverA", "DriverB"],
        "Type": ["stock", "constant", "constant"],
        "Tags": [0, 1, 1],
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({"From": ["DriverA", "DriverB"], "Type": ["+", "+"],
                         "To": ["Outcome", "Outcome"]})
    with pd.ExcelWriter(path) as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)


def write_equation_model(path):
    """Auxiliary outcome with a custom equation whose 2nd parameter is scaled down 100x."""
    df_e = pd.DataFrame({
        "Label": ["Outcome", "DriverA", "DriverB"],
        "Type": ["auxiliary", "constant", "constant"],
        "Tags": [0, 1, 1],
        "Description": ["VOI", "", ""],
        "Equation": ["#1 * DriverA + 0.01 * #2 * DriverB", None, None],
    })
    df_c = pd.DataFrame({"From": ["DriverA", "DriverB"], "Type": ["+", "+"],
                         "To": ["Outcome", "Outcome"]})
    with pd.ExcelWriter(path) as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)


def test_run_gsa_stock_model_symmetric(tmp_path):
    """Two symmetric additive drivers each carry ~half the variance."""
    write_stock_model(tmp_path / "stock.xlsx")
    sdm = SDM(_settings(tmp_path / "stock.xlsx"))
    df = sdm.run_GSA(n=256, seed=1, show_progress=False)

    assert list(df.columns) == ["parameter", "S1", "S1_conf", "ST", "ST_conf"]
    assert set(df["parameter"]) == {"Outcome <- DriverA", "Outcome <- DriverB"}
    assert (df["S1"] <= df["ST"] + 0.06).all()
    assert (df["ST"] <= 1.05).all()
    # symmetric -> the two ST values are close
    st = df.set_index("parameter")["ST"]
    assert abs(st["Outcome <- DriverA"] - st["Outcome <- DriverB"]) < 0.15


def test_run_gsa_equation_model_identifies_dominant_param(tmp_path):
    """The down-scaled parameter (#2) contributes far less variance than #1."""
    write_equation_model(tmp_path / "eq.xlsx")
    sdm = SDM(_settings(tmp_path / "eq.xlsx", pva=0.3))
    df = sdm.run_GSA(n=256, seed=1, show_progress=False)
    top = df.iloc[0]["parameter"]
    assert top.endswith("#1")
    st = df.set_index("parameter")["ST"]
    assert st["Outcome | #1"] > st["Outcome | #2"]


def test_run_gsa_fixed_parameters_excluded(tmp_path):
    """A parameter pinned via bounds (low == high) is dropped from the indices and reported."""
    write_stock_model(tmp_path / "stock.xlsx")
    sdm = SDM(_settings(tmp_path / "stock.xlsx"))
    df = sdm.run_GSA(n=128, seed=1, show_progress=False,
                     bounds={"Outcome <- DriverB": (0.1, 0.1)})
    assert set(df["parameter"]) == {"Outcome <- DriverA"}
    assert "Outcome <- DriverB" in df.attrs["fixed_parameters"]


def write_two_outcome_model(path):
    """Two stocks, each driven by the shared constant lever, so both have variance."""
    df_e = pd.DataFrame({
        "Label": ["Out1", "Out2", "Driver"],
        "Type": ["stock", "stock", "constant"],
        "Tags": [0, 0, 1],
        "Description": ["VOI", "VOI", ""],
    })
    df_c = pd.DataFrame({"From": ["Driver", "Driver"], "Type": ["+", "+"],
                         "To": ["Out1", "Out2"]})
    with pd.ExcelWriter(path) as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)


def test_run_gsa_multiple_vois_returns_dict(tmp_path):
    write_two_outcome_model(tmp_path / "two.xlsx")
    sdm = SDM(_settings(tmp_path / "two.xlsx"))
    res = sdm.run_GSA(["Out1", "Out2"], n=128, seed=1, show_progress=False)
    assert isinstance(res, dict)
    assert set(res) == {"Out1", "Out2"}
    assert all(isinstance(v, pd.DataFrame) for v in res.values())


def test_plot_gsa_returns_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sip_systemsinsightpipeline.plots import plot_gsa

    write_stock_model(tmp_path / "stock.xlsx")
    sdm = SDM(_settings(tmp_path / "stock.xlsx"))
    df = sdm.run_GSA(n=128, seed=1, show_progress=False)
    fig = plot_gsa(df)
    assert fig is not None
    assert len(fig.axes) >= 1

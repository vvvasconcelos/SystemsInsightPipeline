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


def test_clip_enforces_unit_interval_and_asymmetric_ci():
    """clip=True projects a true-zero index (Ishigami x3) onto [0,1] with an asymmetric CI;
    clip=False preserves the raw (possibly negative) estimate as a diagnostic."""
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    names = ["x1", "x2", "x3"]
    bounds = [(-np.pi, np.pi)] * 3
    f = lambda r: Ishigami.evaluate(r.reshape(1, -1))[0]

    clipped, _, _ = gsa.run_sobol(names, bounds, f, n=128, seed=3, clip=True)
    # every index and every lower CI bound is within [0, 1]
    assert (clipped["S1"] >= 0).all() and (clipped["S1"] <= 1).all()
    assert (clipped["S1_low"] >= 0).all() and (clipped["ST_low"] >= 0).all()
    assert (clipped["S1_low"] <= clipped["S1"] + 1e-9).all()
    assert (clipped["S1"] <= clipped["S1_high"] + 1e-9).all()
    # the true-zero parameter clips to exactly 0 with a one-sided interval
    x3 = clipped.set_index("parameter").loc["x3"]
    assert x3["S1"] == 0.0
    assert x3["S1_low"] == 0.0 and x3["S1_high"] > 0.0

    raw, _, _ = gsa.run_sobol(names, bounds, f, n=128, seed=3, clip=False)
    # at this small n the raw estimator for x3 dips below zero (the diagnostic clip removes)
    assert raw.set_index("parameter").loc["x3", "S1"] < 0.0


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

    for col in ("parameter", "S1", "S1_low", "S1_high", "S1_conf", "ST", "ST_low", "ST_high", "ST_conf"):
        assert col in df.columns
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


# --------------------------------------------------------------------------- BCa intervals

def test_bca_interval_respects_boundary_and_is_asymmetric():
    """BCa intervals for a true-zero index stay within [0,1], are asymmetric, never below 0,
    and differ from the plain percentile interval."""
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    names = ["x1", "x2", "x3"]
    bounds = [(-np.pi, np.pi)] * 3
    f = lambda r: Ishigami.evaluate(r.reshape(1, -1))[0]

    bca, _, _ = gsa.run_sobol(names, bounds, f, n=256, seed=3, ci="bca")
    pct, _, _ = gsa.run_sobol(names, bounds, f, n=256, seed=3, ci="percentile")

    assert (bca["S1_low"] >= 0).all() and (bca["ST_low"] >= 0).all()
    assert (bca["S1_high"] <= 1.0 + 1e-9).all()
    x3 = bca.set_index("parameter").loc["x3"]
    assert x3["S1_low"] == 0.0 and x3["S1_high"] > x3["S1_low"]   # one-sided / asymmetric
    # BCa and percentile give genuinely different bounds (the bias/skew correction does something)
    merged = bca.set_index("parameter")["S1_high"] - pct.set_index("parameter")["S1_high"]
    assert (merged.abs() > 1e-6).any()


def test_sobol_estimator_matches_salib():
    """Our vectorised _sobol_point reproduces SALib's first_order/total_order exactly."""
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    sa = pytest.importorskip("SALib.analyze.sobol")
    ss = pytest.importorskip("SALib.sample.sobol")
    prob = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}
    X = ss.sample(prob, 128, calc_second_order=False, seed=1)
    Y = Ishigami.evaluate(X)
    N = len(Y) // 5
    A, B, AB, _ = sa.separate_output_values(Y, 3, N, False)
    s1, st = gsa._sobol_point(A, B, AB)
    salib_s1 = np.array([sa.first_order(A, AB[:, j], B) for j in range(3)]).ravel()
    salib_st = np.array([sa.total_order(A, AB[:, j], B) for j in range(3)]).ravel()
    assert np.allclose(s1, salib_s1)
    assert np.allclose(st, salib_st)


# --------------------------------------------------------------------------- moment-independent

def test_delta_captures_distributional_effect_missed_by_variance():
    """Borgonovo delta gives Ishigami x3 a clearly positive importance, while its variance-based
    first-order S1 ~ 0 -- the distributional effect variance decomposition misses."""
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    latin = pytest.importorskip("SALib.sample.latin")
    names = ["x1", "x2", "x3"]
    prob = {"num_vars": 3, "names": names, "bounds": [[-np.pi, np.pi]] * 3}
    X = latin.sample(prob, 3000, seed=1)
    Y = Ishigami.evaluate(X)

    df = gsa.moment_independent(X, Y, names, method="delta", seed=1)
    assert list(df.columns) == ["parameter", "delta", "delta_conf", "S1", "S1_conf"]
    assert (df["delta"] >= 0).all()                          # non-negative by construction
    d = df.set_index("parameter")
    assert d.loc["x3", "delta"] > 0.08                       # delta sees x3
    assert d.loc["x3", "S1"] < 0.05                          # variance-based first-order does not
    # x1, x2 (the main-effect drivers) rank above x3
    assert d.loc["x1", "delta"] > d.loc["x3", "delta"]
    assert d.loc["x2", "delta"] > d.loc["x3", "delta"]


def test_pawn_nonnegative_and_ranks_drivers():
    Ishigami = pytest.importorskip("SALib.test_functions.Ishigami")
    latin = pytest.importorskip("SALib.sample.latin")
    names = ["x1", "x2", "x3"]
    prob = {"num_vars": 3, "names": names, "bounds": [[-np.pi, np.pi]] * 3}
    X = latin.sample(prob, 3000, seed=2)
    Y = Ishigami.evaluate(X)
    df = gsa.moment_independent(X, Y, names, method="pawn", seed=2)
    assert "pawn_median" in df.columns
    assert (df["pawn_median"] >= 0).all()
    m = df.set_index("parameter")["pawn_median"]
    assert m["x2"] > m["x3"] and m["x1"] > m["x3"]


def test_moment_independent_reproducible_and_degenerate():
    names = ["a", "b"]
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(500, 2))
    y = X[:, 0] + 0.01 * rng.standard_normal(500)
    a = gsa.moment_independent(X, y, names, method="delta", seed=5)
    b = gsa.moment_independent(X, y, names, method="delta", seed=5)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="zero variance"):
        gsa.moment_independent(X, np.ones(500), names, method="delta")


def test_run_gsa_delta_and_pawn_on_model(tmp_path):
    """SDM.run_GSA(method='delta'/'pawn') runs end to end and ranks the dominant equation
    parameter above the down-scaled one."""
    df_e = pd.DataFrame({
        "Label": ["Outcome", "DriverA", "DriverB"],
        "Type": ["auxiliary", "constant", "constant"],
        "Tags": [0, 1, 1],
        "Description": ["VOI", "", ""],
        "Equation": ["#1 * DriverA + 0.01 * #2 * DriverB", None, None],
    })
    df_c = pd.DataFrame({"From": ["DriverA", "DriverB"], "Type": ["+", "+"],
                         "To": ["Outcome", "Outcome"]})
    with pd.ExcelWriter(tmp_path / "eq.xlsx") as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)
    s = Extract(str(tmp_path / "eq.xlsx")).extract_settings()
    s.seed, s.N, s.t_end = 0, 1, 10
    s.parameter_value_aux = s.parameter_value_stocks = 0.3
    sdm = SDM(s)

    dl = sdm.run_GSA(method="delta", n=400, seed=1, show_progress=False)
    assert dl.iloc[0]["parameter"].endswith("#1")
    assert (dl["delta"] >= 0).all()
    pw = sdm.run_GSA(method="pawn", n=400, seed=1, show_progress=False)
    assert pw.iloc[0]["parameter"].endswith("#1")
    assert (pw["pawn_median"] >= 0).all()

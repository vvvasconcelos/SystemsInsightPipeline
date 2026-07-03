"""Tests for plots.plot_trajectories (the time-course plotting API)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline import SDM, Extract
from sip_systemsinsightpipeline.plots import plot_trajectories


@pytest.fixture()
def simulated_sdm(tmp_path):
    df_e = pd.DataFrame({
        "Label": ["S", "LeverA", "LeverB"],
        "Type": ["stock", "constant", "constant"],
        "Tags": [0, 1, 1],
        "Description": ["VOI", "", ""],
    })
    df_c = pd.DataFrame({"From": ["LeverA", "LeverB"], "Type": ["+", "+"], "To": ["S", "S"]})
    with pd.ExcelWriter(tmp_path / "m.xlsx") as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)
    s = Extract(str(tmp_path / "m.xlsx")).extract_settings()
    s.seed = 0
    s.N = 6
    s.t_end = 10
    s.time_unit = "years"
    s.parameter_value_aux = 0.3
    s.parameter_value_stocks = 0.3
    sdm = SDM(s)
    sdm.t_eval = np.linspace(0, s.t_end, 21)
    sdm.run_simulations()
    return s, sdm


def test_band_default_voi_first_intervention(simulated_sdm):
    s, sdm = simulated_sdm
    fig = plot_trajectories(sdm)
    assert fig is not None and len(fig.axes) >= 1
    ax = fig.axes[0]
    assert "years" in ax.get_xlabel()
    # the band plot draws exactly one median line per selected intervention (default: 1)
    assert len(ax.lines) >= 1


def test_spaghetti_multiple_interventions_and_variables(simulated_sdm):
    s, sdm = simulated_sdm
    fig = plot_trajectories(sdm, variables=["S", "LeverA"], interventions="all",
                            kind="spaghetti", max_spaghetti=5)
    # one panel per variable
    visible = [a for a in fig.axes if a.get_visible() and a.axison]
    assert len(visible) == 2


def test_unknown_intervention_raises(simulated_sdm):
    s, sdm = simulated_sdm
    with pytest.raises(ValueError, match="not an intervention"):
        plot_trajectories(sdm, interventions="NotALever")


def test_requires_run_simulations(tmp_path, simulated_sdm):
    s, _ = simulated_sdm
    fresh = SDM(s)
    with pytest.raises(ValueError, match="run_simulations"):
        plot_trajectories(fresh)


def test_warns_on_two_point_t_eval(tmp_path):
    df_e = pd.DataFrame({
        "Label": ["S", "Lever"], "Type": ["stock", "constant"],
        "Tags": [0, 1], "Description": ["VOI", ""],
    })
    df_c = pd.DataFrame({"From": ["Lever"], "Type": ["+"], "To": ["S"]})
    with pd.ExcelWriter(tmp_path / "m.xlsx") as w:
        df_e.to_excel(w, sheet_name="Elements", index=False)
        df_c.to_excel(w, sheet_name="Connections", index=False)
    s = Extract(str(tmp_path / "m.xlsx")).extract_settings()
    s.seed, s.N, s.t_end, s.time_unit = 0, 3, 10, "years"
    s.parameter_value_aux = s.parameter_value_stocks = 0.3
    sdm = SDM(s)                       # default t_eval: endpoints only
    sdm.run_simulations()
    with pytest.warns(UserWarning, match="recorded time points"):
        plot_trajectories(sdm)

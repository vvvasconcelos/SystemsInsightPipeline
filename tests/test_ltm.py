"""Tests for Loops That Matter (sip_systemsinsightpipeline.ltm)."""
import numpy as np
import pandas as pd
import pytest

from sip_systemsinsightpipeline import LoopsThatMatter, find_feedback_loops
from sip_test_helpers import NAN, fixed_params, load_model, quiet, write_workbook


def adjacency(names, links):
    df = pd.DataFrame(0.0, index=names, columns=names)
    for src, tgt in links:
        df.loc[tgt, src] = 1.0
    return df


def test_find_feedback_loops_is_canonical():
    names = ["A", "B", "C", "D"]
    # loops: A->B->A, B->C->D->B, C->C (self loop); edges listed in a scrambled order
    df = adjacency(names, [("D", "B"), ("C", "D"), ("B", "A"), ("C", "C"), ("A", "B"), ("B", "C")])
    loops = find_feedback_loops(df, stocks=names)
    assert loops == [["C"], ["A", "B"], ["B", "C", "D"]]          # by length, then model order
    assert find_feedback_loops(df, stocks=names, max_length=2) == [["C"], ["A", "B"]]
    assert find_feedback_loops(df, stocks=["D"]) == [["B", "C", "D"]]
    assert find_feedback_loops(df, stocks=[], require_stock=False) == loops


def test_loop_names_and_polarity(tmp_path):
    wb = write_workbook(tmp_path / "loops.xlsx",
                        {"Label": ["S", "A", "K"], "Type": ["stock", "auxiliary", "constant"],
                         "Tags": [0, 0, 1], "Description": ["VOI", "", ""]},
                        [("S", "-", "S"), ("S", "+", "A"), ("A", "+", "S"), ("K", "+", "A")])
    s, sdm = load_model(wb)
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    assert ltm.feedback_loops == [["S"], ["S", "A"]]
    assert ltm.loop_names == ["B1", "R1"]


def test_equation_stock_link_scores_use_the_custom_equation(tmp_path):
    """dS/dt = A - 0.2 S with A = S + K: the stock's own equation drives its link scores."""
    wb = write_workbook(tmp_path / "eqstock.xlsx",
                        {"Label": ["S", "A", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                         "Description": ["VOI", "", ""], "Equation": ["0.5 * A - 0.2 * S", "S + K", NAN]},
                        [("A", "+", "S"), ("S", "+", "A"), ("K", "+", "A")])
    s, sdm = load_model(wb, t_end=2.0)
    params = fixed_params(sdm)
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    (df, links, loops), _, _ = quiet(ltm.run_ltm_analysis, params, n_points=11,
                                     intervention_intensities=[1.0], progress_bar=False)
    a_to_s = links[(links.source == "A") & (links.target == "S")]
    # delta_target of a stock target is the change in its net flow, computed from the equation
    s_vals, a_vals = df["S"].to_numpy(), df["A"].to_numpy()
    rhs = 0.5 * a_vals - 0.2 * s_vals
    assert np.allclose(a_to_s["delta_target"].to_numpy(), np.diff(rhs))
    assert np.allclose(a_to_s["delta_z_x"].to_numpy(), 0.5 * np.diff(a_vals))


def test_time_dependent_equations_in_ltm(tmp_path):
    wb = write_workbook(tmp_path / "time.xlsx",
                        {"Label": ["S", "A", "K"], "Type": ["stock", "auxiliary", "constant"], "Tags": [0, 0, 1],
                         "Description": ["VOI", "", ""], "Equation": ["A - 0.1 * S", "(t > 1) * K + 0.1 * S", NAN]},
                        [("A", "+", "S"), ("S", "+", "A"), ("K", "+", "A")])
    s, sdm = load_model(wb, t_end=3.0)
    ltm, _, _ = quiet(LoopsThatMatter, sdm)
    (_, links, loops), _, _ = quiet(ltm.run_ltm_analysis, fixed_params(sdm), n_points=31,
                                    intervention_intensities=[1.0], progress_bar=False)
    assert np.isfinite(links["ltm_link_score"]).all()
    assert set(loops.columns) >= {"time", "R1_raw", "R1_relative"}


def test_ltm_names_are_deterministic_within_a_process(tmp_path, tutorials_dir):
    s, sdm = load_model(tutorials_dir / "Insulation_Decisions.xlsx", pv=0.1)
    a, _, _ = quiet(LoopsThatMatter, sdm)
    b, _, _ = quiet(LoopsThatMatter, sdm)
    assert a.feedback_loops == b.feedback_loops and a.loop_names == b.loop_names
    lengths = [len(loop) for loop in a.feedback_loops]
    assert lengths == sorted(lengths)

"""Plot layout: titles and axis labels stay inside the figure."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sip_systemsinsightpipeline.plots import plot_simulated_intervention_ranking

LONG_OUTCOME = "Percentage of electricity use during peak hours (16.00-21.00) on household level"


def _x_extent(artist, fig):
    """Horizontal extent of a drawn artist, in inches from the figure's left edge."""
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.x0 / fig.dpi, box.x1 / fig.dpi


@pytest.fixture
def ranking_figure():
    s = SimpleNamespace(t_end=15, time_unit="years",
                        intervention_strengths={"Lever A": 1.0, "Lever B": -1.0})
    rng = np.random.default_rng(0)
    effects = {"Lever A": rng.normal(1.0, 1.0, 20), "Lever B": rng.normal(-2.0, 1.0, 20)}
    fig = plot_simulated_intervention_ranking(s, effects, LONG_OUTCOME)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def test_ranking_default_xlabel_fits_under_the_axes(ranking_figure):
    ax = ranking_figure.axes[0]
    assert ax.get_xlabel() == "Effect after 15 years\n(vs. no-intervention baseline, model units)"
    axes_x0, axes_x1 = _x_extent(ax.patch, ranking_figure)
    label_x0, label_x1 = _x_extent(ax.xaxis.label, ranking_figure)
    assert axes_x0 <= label_x0 and label_x1 <= axes_x1


def test_ranking_long_title_is_wrapped_inside_the_figure(ranking_figure):
    ax = ranking_figure.axes[0]
    assert ax.get_title().replace("\n", " ") == "Effect on " + LONG_OUTCOME
    title_x0, title_x1 = _x_extent(ax.title, ranking_figure)
    assert 0 <= title_x0 and title_x1 <= ranking_figure.get_figwidth()


def test_ranking_top_plot_counts_only_other_interventions():
    # The outcome A is also a lever, with the largest effect; it is not drawn, and top_plot=2
    # still shows two interventions.
    s = SimpleNamespace(t_end=10, time_unit="years",
                        intervention_strengths={"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    effects = {"A": np.full(5, 9.0), "B": np.full(5, 3.0), "C": np.full(5, 2.0),
               "D": np.full(5, 1.0)}
    fig = plot_simulated_intervention_ranking(s, effects, "A", top_plot=2)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["B (+1.0)", "C (+1.0)"]
    plt.close(fig)
    fig = plot_simulated_intervention_ranking(s, effects, "A", order=["D", "A", "B"])
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["D (+1.0)", "B (+1.0)"]
    plt.close(fig)


def test_ranking_xlabel_can_be_replaced():
    s = SimpleNamespace(t_end=10, time_unit="years", intervention_strengths={"B": 1.0})
    fig = plot_simulated_intervention_ranking(s, {"B": np.array([1.0, 2.0, 3.0])}, "A",
                                              xlabel="Final value of A")
    assert fig.axes[0].get_xlabel() == "Final value of A"
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import pandas as pd
sns.set_theme()


def _wrap_param_label(name, width=40):
    """Format a sensitivity parameter name into a readable (possibly two-line) tick label.

    Parameter names produced by the GSA machinery look like ``"TARGET <- SOURCE"`` (a causal
    link) or ``"TARGET | #k"`` (a custom-equation parameter). Long names are split across two
    lines so that *both* halves stay legible: the target on the first line and ``← source`` on
    the second, each truncated to ``width`` characters. Plain names are returned as-is (or
    truncated). Use a small font (see ``label_fontsize``) alongside this.
    """
    name = str(name)

    def cut(s):
        s = s.strip()
        return s if len(s) <= width else s[: width - 1] + "…"

    for sep, joiner in (("<-", "←"), ("|", "|")):
        if sep in name:
            head, tail = name.split(sep, 1)
            return f"{cut(head)}\n{joiner} {cut(tail)}"
    return cut(name)

def plot_trajectories(sdm, variables=None, interventions=None, kind="band",
                      percentiles=(10, 90), max_spaghetti=40, ncols=2, figsize=None,
                      label_fontsize=9):
    """Plot simulated trajectories of model variables over time.

    Requires :meth:`SDM.run_simulations` to have been called. Produces one panel per
    variable; within each panel, one colour per selected intervention run.

    The *shape* of these curves is often the most informative output of a system dynamics
    model: exponential growth (a dominant reinforcing loop), goal-seeking (balancing loop),
    S-shaped growth (reinforcing then balancing), overshoot-and-collapse, or oscillation
    (balancing loop with delay). See docs/trajectories-and-archetypes.html for a guide.

    Parameters
    ----------
    sdm : SDM
        A simulated model (``sdm.run_simulations()`` already called).
    variables : str or list of str, optional
        Which variables to plot (stocks, auxiliaries or constants). Defaults to the
        variable(s) of interest.
    interventions : str or list of str, optional
        Which intervention runs to show. Defaults to the first intervention variable;
        pass a list to compare several, or ``"all"`` for every intervention (capped at 6
        with a warning, to keep the plot readable).
    kind : {"band", "spaghetti"}
        ``"band"``: median across the N parameter samples with a percentile band
        (default 10th-90th). ``"spaghetti"``: individual sample trajectories (up to
        ``max_spaghetti`` per intervention), which better reveals qualitatively different
        behaviours hidden by the median.
    percentiles : (low, high)
        Band percentiles for ``kind="band"``.

    Returns the matplotlib Figure. matplotlib-only; does not set a global style.

    Notes
    -----
    By default the solver only *records* the first and last time point. For smooth curves,
    set ``sdm.t_eval = np.linspace(0, s.t_end, 50)`` **before** calling ``run_simulations``.
    """
    import textwrap
    import warnings as _warnings

    if getattr(sdm, "df_sol_per_sample", None) is None:
        raise ValueError("Call sdm.run_simulations() before plotting trajectories.")
    t = np.asarray(sdm.t_eval, dtype=float)
    if t.size < 3:
        _warnings.warn(
            "Only {} recorded time points - trajectories will be straight lines. Set "
            "sdm.t_eval = np.linspace(0, t_end, 50) before run_simulations() for smooth curves.".format(t.size))

    voi = sdm.variable_of_interest
    if variables is None:
        variables = list(voi) if isinstance(voi, (list, tuple)) else [voi]
    elif isinstance(variables, str):
        variables = [variables]

    all_ints = list(sdm.intervention_variables)
    if interventions is None:
        interventions = [all_ints[0]]
    elif interventions == "all":
        interventions = all_ints
        if len(interventions) > 6:
            _warnings.warn(f"{len(interventions)} interventions; showing the first 6. "
                           "Pass an explicit list to choose which.")
            interventions = interventions[:6]
    elif isinstance(interventions, str):
        interventions = [interventions]
    for iv in interventions:
        if iv not in all_ints:
            raise ValueError(f"'{iv}' is not an intervention variable.")

    n_panels = len(variables)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (5.6 * ncols, 3.8 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axs.flatten()

    colors = plt.get_cmap("tab10")
    N = len(sdm.df_sol_per_sample)

    for p, var in enumerate(variables):
        ax = axes[p]
        for ci, iv in enumerate(interventions):
            i = all_ints.index(iv)
            series = np.array([sdm.df_sol_per_sample[n][i][var].values for n in range(N)], dtype=float)
            short_iv = iv if len(iv) <= 32 else iv[:29] + "..."
            if kind == "band":
                med = np.median(series, axis=0)
                lb = np.percentile(series, percentiles[0], axis=0)
                ub = np.percentile(series, percentiles[1], axis=0)
                ax.plot(t, med, color=colors(ci), lw=2.0, label=short_iv, zorder=3)
                ax.fill_between(t, lb, ub, color=colors(ci), alpha=0.15, lw=0, zorder=2)
            elif kind == "spaghetti":
                shown = min(N, max_spaghetti)
                for n in range(shown):
                    ax.plot(t, series[n], color=colors(ci), alpha=max(0.08, 1.5 / shown),
                            lw=0.9, zorder=2)
                ax.plot([], [], color=colors(ci), label=short_iv)  # legend proxy
            else:
                raise ValueError(f"kind must be 'band' or 'spaghetti', got {kind!r}.")
        ax.axhline(0, color="#888888", lw=0.7, ls=":", zorder=1)
        ax.set_title("\n".join(textwrap.wrap(var, 46)), fontsize=label_fontsize + 1)
        time_unit = getattr(getattr(sdm, "s", None), "time_unit", None) or "time"
        ax.set_xlabel(f"Time ({time_unit})", fontsize=label_fontsize)
        ax.tick_params(labelsize=label_fontsize)
        if len(interventions) > 1 and p == 0:
            ax.legend(fontsize=max(7, label_fontsize - 1), title="Intervention on",
                      title_fontsize=max(7, label_fontsize - 1))
    for q in range(n_panels, len(axes)):
        axes[q].axis("off")
    fig.tight_layout()
    return fig

def plot_simulated_intervention_ranking(s, intervention_effects, voi, top_plot=None, order=None):
    """ Plot simulated intervention effects in a horizontal boxplot, ranked by median.
    """
    df_SA = pd.DataFrame(intervention_effects)
    df_SA = df_SA.reindex(columns=list(
        df_SA.abs().median().sort_values(ascending=False).index))

    if top_plot is not None:
        df_SA = df_SA[list(df_SA.columns)[:top_plot]]  # Take only the top X interventions to plot

    if order is not None:
        df_SA = df_SA[order]

    if voi in df_SA.columns:
        df_SA = df_SA.drop(voi, axis=1)  # Remove the variable of interest as intervention variable from the plot

    name_with_intervention = []

    for name in list(df_SA.columns):
        if "+" in name:
            name_1, name_2 = name.split("+")
            # Format for name_1
            effect_1 = s.intervention_strengths[name_1]
            formatted_1 = " ".join(name_1.split("_")) + f" ({'+' if effect_1 > 0 else ''}{effect_1})"
            # Format for name_2
            effect_2 = s.intervention_strengths[name_2]
            formatted_2 = " ".join(name_2.split("_")) + f" ({'+' if effect_2 > 0 else ''}{effect_2})"
            name_with_intervention += [formatted_1 + ' & ' + formatted_2]
        else:
            effect = s.intervention_strengths[name]
            formatted = " ".join(name.split("_")) + f" ({'+' if effect > 0 else ''}{effect})"
            name_with_intervention.append(formatted)
                
    df_SA = df_SA.rename(mapper=dict(zip(df_SA.columns, name_with_intervention)), axis=1)

    # Define a consistent color palette for variables
    unique_vars = df_SA.columns
    palette_dict = {var: sns.color_palette("husl", len(unique_vars))[i] for i, var in enumerate(unique_vars)}
    palette = [palette_dict[var] for var in df_SA.columns]

    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111)
    sns.boxplot(data=df_SA, showfliers=False, whis=True, orient='h', palette=palette)
    plt.vlines(x=0, ymin=-0.5, ymax=len(df_SA.columns) - 0.6, colors='black', linestyles='dashed')
    plt.title("Effect on " + " ".join(voi.split("_")))
    # Effects are the raw change in the outcome at the final time point (model units),
    # not standardized scores.
    plt.xlabel("Effect after " + str(s.t_end) + " " + s.time_unit + " (final-time value, model units)")
    plt.ylabel("")
    return fig


def plot_gsa(gsa_df, kind="tornado", top=None, title="Global sensitivity (Sobol indices)", ax=None,
             wrap=True, label_width=40, label_fontsize=8):
    """Sobol tornado: horizontal bars of total-order index ST with first-order S1 overlaid.

    Parameters
    ----------
    gsa_df : pandas.DataFrame
        Output of :meth:`SDM.run_GSA`. Uses the asymmetric ``*_low`` / ``*_high`` confidence
        bounds when present (falling back to the symmetric ``*_conf`` half-width otherwise).
    kind : str
        Only ``"tornado"`` is currently supported (kept for forward compatibility).
    top : int, optional
        Show only the ``top`` parameters by ST (the rest are omitted).
    ax : matplotlib Axes, optional
        Draw onto an existing axes; a new figure is created otherwise.
    wrap : bool
        Split long ``"TARGET <- SOURCE"`` parameter names across two lines so both halves stay
        legible (see :func:`_wrap_param_label`); ``label_width`` truncates each half and
        ``label_fontsize`` sets the tick font size.

    Returns the matplotlib Figure. matplotlib-only; does not set a global style.
    """
    if kind != "tornado":
        raise ValueError(f"Unsupported kind {kind!r}; only 'tornado' is available.")

    df = gsa_df.sort_values("ST", ascending=False)
    if top is not None:
        df = df.head(int(top))
    # Plot ascending so the most influential parameter sits at the top.
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))

    labels = [_wrap_param_label(p, label_width) if wrap else str(p) for p in df["parameter"]]
    row_h = 0.62 if wrap else 0.45   # two-line labels need more vertical room
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(2.5, row_h * len(df) + 1.6)))
    else:
        fig = ax.figure

    def asym_err(point_col, low_col, high_col, conf_col):
        """Return a 2xN [lower, upper] error array from asymmetric bounds if available."""
        point = df[point_col].values
        if low_col in df.columns and high_col in df.columns:
            lower = np.clip(point - df[low_col].values, 0, None)
            upper = np.clip(df[high_col].values - point, 0, None)
            return np.vstack([lower, upper])
        half = df[conf_col].values
        return np.vstack([half, half])

    st_color = "#9ecae1"
    s1_color = "#08519c"
    ax.barh(y, df["ST"], height=0.6, color=st_color, edgecolor="white",
            xerr=asym_err("ST", "ST_low", "ST_high", "ST_conf"),
            error_kw=dict(ecolor="#5a6b7b", lw=1, capsize=3),
            label="ST  (total effect, incl. interactions)", zorder=2)
    ax.errorbar(df["S1"], y, xerr=asym_err("S1", "S1_low", "S1_high", "S1_conf"),
                fmt="o", color=s1_color, markersize=5, lw=1, capsize=3,
                label="S1  (first-order effect)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("Sobol sensitivity index")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    ax.axvline(0, color="#888888", lw=0.8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.margins(y=0.02)
    fig.tight_layout()
    return fig


def plot_moment_independent(df, measure="delta", top=None, title=None, ax=None,
                            wrap=True, label_width=40, label_fontsize=8):
    """Horizontal bar chart of a moment-independent sensitivity measure.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :meth:`SDM.run_GSA` with ``method="delta"`` (columns ``delta, delta_conf``)
        or ``method="pawn"`` (column ``pawn_median``).
    measure : {"delta", "pawn"}
        Which column to plot.
    top : int, optional
        Show only the ``top`` parameters by the measure.
    wrap : bool
        Split long ``"TARGET <- SOURCE"`` names across two lines (see ``plot_gsa``).

    Returns the matplotlib Figure. matplotlib-only; does not set a global style.
    """
    if measure == "delta":
        value_col, conf_col, sort_col, label = "delta", "delta_conf", "delta", "Borgonovo δ (moment-independent)"
        default_title = "Distributional importance (Borgonovo δ)"
    elif measure == "pawn":
        value_col, conf_col, sort_col, label = "pawn_median", None, "pawn_median", "PAWN (median KS statistic)"
        default_title = "Distributional importance (PAWN)"
    else:
        raise ValueError(f"measure must be 'delta' or 'pawn', got {measure!r}.")
    if value_col not in df.columns:
        raise ValueError(f"DataFrame has no '{value_col}' column; was it produced with method='{measure}'?")

    d = df.sort_values(sort_col, ascending=False)
    if top is not None:
        d = d.head(int(top))
    d = d.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))

    labels = [_wrap_param_label(p, label_width) if wrap else str(p) for p in d["parameter"]]
    row_h = 0.62 if wrap else 0.45
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(2.5, row_h * len(d) + 1.6)))
    else:
        fig = ax.figure

    xerr = d[conf_col].values if conf_col and conf_col in d.columns else None
    ax.barh(y, d[value_col], height=0.6, color="#74a9cf", edgecolor="white",
            xerr=xerr, error_kw=dict(ecolor="#5a6b7b", lw=1, capsize=3), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel(label)
    ax.set_title(title or default_title, loc="left", fontweight="bold")
    ax.set_xlim(left=0)
    ax.margins(y=0.02)
    fig.tight_layout()
    return fig


def plot_scenario_tradeoff(result, ax=None, title="Scenario discovery: coverage-density trade-off"):
    """Plot the PRIM peeling trajectory in coverage-density space.

    Each point is one peel step; peeling trades coverage (recall) for density (precision).
    The selected (recommended) box is highlighted. matplotlib-only.
    """
    if result.trajectory is None:
        raise ValueError("No peeling trajectory; plot_scenario_tradeoff applies to PRIM results.")
    traj = result.trajectory
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(traj["coverage"], traj["density"], "-o", color="#9ecae1",
            markersize=4, lw=1.5, label="peeling trajectory", zorder=2)
    chosen = result.box
    ax.scatter([chosen.coverage], [chosen.density], s=110, color="#08519c",
               edgecolor="white", zorder=3, label="recommended box")
    ax.annotate(f"density={chosen.density:.2f}\ncoverage={chosen.coverage:.2f}",
                (chosen.coverage, chosen.density), textcoords="offset points",
                xytext=(8, -22), fontsize=9, color="#08519c")
    ax.set_xlabel("Coverage  (recall — share of interesting cases captured)")
    ax.set_ylabel("Density  (precision — share of box that is interesting)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(traj["density"].min(), 0) - 0.02, 1.02)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig


def plot_scenario_box(result, x_dim, y_dim, X, box=None, ax=None,
                      title="Scenario discovery: input space"):
    """Scatter two chosen inputs coloured by interesting/not, with the selected box drawn.

    ``result`` is a ScenarioResult; ``X`` the input DataFrame used to build it. ``box`` defaults
    to ``result.box``. Only the ``x_dim``/``y_dim`` edges of the box are drawn (a box may
    restrict other dimensions too). matplotlib-only.
    """
    import matplotlib.patches as mpatches

    box = box if box is not None else result.box
    mask = np.asarray(result.mask, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure

    ax.scatter(X[x_dim].values[~mask], X[y_dim].values[~mask], s=12, alpha=0.35,
               color="#b0b8c1", label="not interesting", zorder=1)
    ax.scatter(X[x_dim].values[mask], X[y_dim].values[mask], s=14, alpha=0.7,
               color="#c0392b", label="interesting", zorder=2)

    lim = box.limits
    x_lo = lim.loc[x_dim, "min"] if x_dim in lim.index else X[x_dim].min()
    x_hi = lim.loc[x_dim, "max"] if x_dim in lim.index else X[x_dim].max()
    y_lo = lim.loc[y_dim, "min"] if y_dim in lim.index else X[y_dim].min()
    y_hi = lim.loc[y_dim, "max"] if y_dim in lim.index else X[y_dim].max()
    ax.add_patch(mpatches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                 fill=False, edgecolor="#08519c", lw=2.2, zorder=3, label="selected box"))

    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig

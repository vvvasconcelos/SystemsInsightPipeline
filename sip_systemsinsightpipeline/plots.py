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

def plot_simulated_interventions(s, df_sol_per_sample, intervention_effects, interval_type="percentile", confidence_bounds=.95, top_plot=None):
    """
    Plot the simulated interventions over time
    """
    if top_plot != None:
        top_plot = min(top_plot or len(intervention_effects), len(intervention_effects))
    
    df_sol_per_sample_reordered = df_sol_per_sample.copy()
    top_vars_plot = list(intervention_effects.keys())[:top_plot]

    for i in range(s.N):
        df_sol_per_sample_dict_i = dict(zip(s.intervention_variables, df_sol_per_sample[i]))
        df_sol_per_sample_reordered[i] = [df_sol_per_sample_dict_i[var] for var in top_vars_plot] 

    df_SA = pd.DataFrame(intervention_effects)
    df_SA = df_SA.reindex(columns=list(
                            df_SA.abs().median().sort_values(ascending=False).index))
    palette_dict = {var : "#4682B4" for var in df_SA.columns}  # Blue for positive effects
    medians = df_SA.median()
    lower_than_zero_vars = medians.loc[medians < 0].index
    for var in lower_than_zero_vars:
        palette_dict[var] = "#FF6347"  # Red for negative effects

    num_plots = len(top_vars_plot)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle("Simulated interventions with N="+ str(s.N) + " samples")
    ax = axs.flatten()

    for k, var in enumerate(top_vars_plot):
        if k >= len(ax):
            break  # Prevent index out of bounds
        if interval_type != "spaghetti":
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in s.t_eval:
                samples_at_time_t = [df_sol_per_sample[n][k].loc[t, s.variable_of_interest] for n in range(s.N)]

                if interval_type == "confidence":
                    label_avg = "Mean"
                    mean = np.mean(samples_at_time_t)
                    standard_error = scipy.stats.sem(samples_at_time_t)
                    h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N-1)
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(mean-h)
                    ub_confs_at_time_t.append(mean+h)

                elif interval_type == "percentile":
                    label_avg = "Median"
                    avg_at_time_t.append(np.median(samples_at_time_t))
                    lower_percentile = (1 - confidence_bounds) / 2 * 100
                    upper_percentile = (1 + confidence_bounds) / 2 * 100
                    lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                    ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
    
            ax[k].plot(s.t_eval, avg_at_time_t, label=label_avg, color=palette_dict[var])
            ax[k].fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                                alpha=.3, label=str(int(confidence_bounds*100)) + "% " + interval_type + " interval", color=palette_dict[var])
        
        else:
            for i, data_i, in enumerate(df_sol_per_sample):
                ax[k].plot(data_i[0].Time, data_i[k][s.variable_of_interest], alpha=.3, color=palette_dict[var])

        label = " ".join(s.variable_of_interest.split("_"))
        ax[k].set_ylabel(label)
        title = " ".join(var.split("_")) #"Intervention on " + " ".join(var.split("_"))
        ax[k].set_title(title)
        #ax[k].set_ylim([min_value, max_value])

        if k >= num_plots - 3:  # Last row of plots
            ax[k].set_xlabel(s.time_unit)

        if k == 0:
            ax[k].legend()

    # Hide unused subplots
    for b in range(num_plots, len(ax)):
        ax[b].axis('off')

    plt.tight_layout()

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
    plt.xlabel("Standardized effect after " + str(s.t_end) + " " + s.time_unit)
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

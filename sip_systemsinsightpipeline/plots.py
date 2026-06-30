import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import pandas as pd
sns.set_theme()

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


def plot_gsa(gsa_df, kind="tornado", top=None, title="Global sensitivity (Sobol indices)", ax=None):
    """Sobol tornado: horizontal bars of total-order index ST with first-order S1 overlaid.

    Parameters
    ----------
    gsa_df : pandas.DataFrame
        Output of :meth:`SDM.run_GSA` — columns ``parameter, S1, S1_conf, ST, ST_conf``.
    kind : str
        Only ``"tornado"`` is currently supported (kept for forward compatibility).
    top : int, optional
        Show only the ``top`` parameters by ST (the rest are omitted).
    ax : matplotlib Axes, optional
        Draw onto an existing axes; a new figure is created otherwise.

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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(df) + 1.5)))
    else:
        fig = ax.figure

    st_color = "#9ecae1"
    s1_color = "#08519c"
    ax.barh(y, df["ST"], height=0.6, color=st_color, edgecolor="white",
            xerr=df["ST_conf"], error_kw=dict(ecolor="#5a6b7b", lw=1, capsize=3),
            label="ST  (total effect, incl. interactions)", zorder=2)
    ax.errorbar(df["S1"], y, xerr=df["S1_conf"], fmt="o", color=s1_color,
                markersize=5, lw=1, capsize=3, label="S1  (first-order effect)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Sobol sensitivity index")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.margins(y=0.02)
    fig.tight_layout()
    return fig

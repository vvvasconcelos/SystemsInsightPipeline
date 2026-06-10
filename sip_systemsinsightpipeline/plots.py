import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
import pandas as pd
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
sns.set_theme()

def plot_simulated_data(s, df_pred, title):
    """
    Plot the simulated data.

    Parameters:
    - df_pred: DataFrame containing the simulated data.
    - title: Title of the plot.
    - s: Object containing system information.

    Returns:
    - None (Displays the plot).
    """

    num_plots = len(s.stocks_and_auxiliaries)
    num_rows = int(np.ceil(num_plots / 3))

    fig, axs = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.suptitle(title)
    color_map = plt.cm.get_cmap('Paired', len(s.stocks_and_auxiliaries))

    ax = axs.flatten()
    N = len(df_pred)

    for k, var in enumerate(s.stocks_and_auxiliaries):
        if s.interval_type != "spaghetti":
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            for t in s.t_eval:
                label_avg = "Mean"
                samples_at_time_t = [df_pred[i].loc[t, var] for i in range(s.N)]

                if s.interval_type == "confidence":
                    mean = np.mean(samples_at_time_t)
                    standard_error = scipy.stats.sem(samples_at_time_t)
                    h = standard_error * scipy.stats.t.ppf((1 + s.confidence_bounds) / 2., N-1)
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(mean-h)
                    ub_confs_at_time_t.append(mean+h)
                elif s.interval_type == "percentile":
                    mean = np.median(samples_at_time_t)
                    lower_percentile = (1 - s.confidence_bounds) / 2 * 100
                    upper_percentile = (1 + s.confidence_bounds) / 2 * 100
                    avg_at_time_t.append(mean)
                    lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                    ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))

            ax[k].plot(s.t_eval, avg_at_time_t, label=label_avg)
            ax[k].fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t,
                               alpha=.3, label=str(int(s.confidence_bounds*100)) + "% CI") #Confidence interval")
        else:
            for i, data_i, in enumerate(df_pred):
                ax[k].plot(data_i.Time, data_i[var], alpha=.3, color=color_map(k)) 
        
        label = " ".join(var.split("_"))
        ax[k].set_ylabel(label)

        if k >= num_plots - 3:  # Last row of plots
            ax[k].set_xlabel(s.time_unit)

        if k == 0:
            ax[k].legend()

    # Hide unused subplot space
    for b in range(num_rows * 3 - num_plots):
        ax[num_plots + b].axis('off')

    plt.tight_layout()
    plt.show()

def plot_simulated_interventions_compare(s, df_sol_per_sample):
    """ Plot the simulated interventions in the same plot.
    """
    # Create widgets
    confidence_bounds_slider = widgets.FloatSlider(value=0.95, min=0.01, max=0.99, step=0.01, description='Interval bounds:')
    variable_selector = widgets.SelectMultiple(options=s.intervention_variables, value=s.intervention_variables[:2], description='Variables:')

    # Update plot function
    def update_plot(confidence_bounds, compare_int_vars):
        plt.figure(figsize=(10, 5))
        color_map = plt.cm.get_cmap('Paired', len(compare_int_vars))
        for k, var in enumerate(compare_int_vars):
            avg_at_time_t = []
            lb_confs_at_time_t = []
            ub_confs_at_time_t = []

            if s.interval_type != "spaghetti":
                for t in s.t_eval:
                    samples_at_time_t = [df_sol_per_sample[n][s.intervention_variables.index(var)].loc[t, s.variable_of_interest] for n in range(s.N)]

                    if s.interval_type ==  "confidence":
                        mean = np.mean(samples_at_time_t)
                        standard_error = scipy.stats.sem(samples_at_time_t)
                        h = standard_error * scipy.stats.t.ppf((1 + confidence_bounds) / 2., s.N-1)
                        avg_at_time_t.append(mean)
                        lb_confs_at_time_t.append(mean-h)
                        ub_confs_at_time_t.append(mean+h)

                    elif s.interval_type ==  "percentile":
                        avg_at_time_t.append(np.median(samples_at_time_t))
                        lower_percentile = (1 - confidence_bounds) / 2 * 100
                        upper_percentile = (1 + confidence_bounds) / 2 * 100
                        lb_confs_at_time_t.append(np.percentile(samples_at_time_t, lower_percentile))
                        ub_confs_at_time_t.append(np.percentile(samples_at_time_t, upper_percentile))
                    
                plt.plot(s.t_eval, avg_at_time_t, label=" ".join(var.split("_"))) 
                plt.fill_between(s.t_eval, lb_confs_at_time_t, ub_confs_at_time_t, alpha=.3)
            else:
                for i, data_i, in enumerate(df_sol_per_sample):
                    plt.plot(data_i[0].Time, data_i[k][s.variable_of_interest], alpha=.3, color=color_map(k))              

        plt.xlabel(s.time_unit)
        plt.ylabel(" ".join(s.variable_of_interest.split("_")))
        plt.legend()
        plt.show()

    return widgets.interactive(update_plot, confidence_bounds=confidence_bounds_slider, compare_int_vars=variable_selector, top_plot=None)
    
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

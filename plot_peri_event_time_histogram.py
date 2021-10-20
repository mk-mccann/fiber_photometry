import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join
from scipy.stats import binned_statistic

import paths
import functions_aggregation as f_aggr
from functions_utils import list_lists_to_array, remove_baseline


def plot_peth(episodes, bin_duration, scoring_type,
              f_trace='zscore', channel_key=None, bin_function=np.nanmedian,
              index_key='overall_episode_number', **kwargs):

    """ Plots a peri-event time histogram of individual episodes of some behavior.

    Parameters
    ----------
    episodes : pd.DataFrame
        pd.DataFrames containing fluorescence data for all episodes of a scoring types
    bin_duration : float, int
        Width of the bins (in seconds) for the histogram
    scoring_type: str
        Name of the episodes being plotting
    f_trace : str, default='zscore'
        The fluorescence trace to be plotted. Options are ['auto', 'gcamp', 'dff', 'zscore'].
    channel_key : str, optional, default=None
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default=None for single-fiber recordings.
    bin_function : optional, default=np.nanmedian
        The function used by scipy.stats.binned_statistic.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        for all options. The user can also specify a custom function.
    index_key : str, default='overall_episode_number'
        Name of the column in 'episodes' used for indexing

    Returns
    -------

    Keyword Arguments
    -----------------
        norm_start : float, int
            Number of seconds before the start of an episode from which to
            calculate baseline for the trace

    See Also
    --------
    scipy.stats.binned_statistic : Bins the data and applies a statistical function to each bin.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html

    """

    # Handle keyword args
    norm_start = kwargs.get('norm_start', -5)
    norm_end = kwargs.get('norm_end', 0)

    if channel_key is None:
        f_trace = f_trace
    else:
        f_trace = '_'.join([f_trace, channel_key])

    # Get episode traces
    times = episodes.groupby([index_key])['normalized_time'].agg(list).to_list()
    traces = episodes.groupby([index_key])[f_trace].agg(list).to_list()

    times = list_lists_to_array(times)
    traces = list_lists_to_array(traces)

    # The times should all be pretty similar (at least close enough for binning)
    # Take an average time trace for our calculate
    time = np.nanmean(times, axis=0)

    # Remove the baseline from the fluorescence traces in the window
    traces = remove_baseline(time, traces, norm_start=norm_start)

    # Bin times
    min_time = np.floor(times[0][0])
    max_time = np.ceil(times[0][-1])
    bins = np.arange(min_time, max_time+bin_duration, bin_duration)

    # Calculate the statistics on the bin
    bin_values, _, _ = binned_statistic(time, traces, statistic=bin_function, bins=bins)

    # Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    cbar_range_max = np.ceil(np.nanmax(traces))
    im = ax.imshow(bin_values, cmap='seismic', vmin=-cbar_range_max, vmax=cbar_range_max)

    x_tick_labels = np.arange(bins[0], bins[-1]+bin_duration, 5)
    x_tick_positions = np.linspace(-0.5, len(bins)-1.5,  len(x_tick_labels))
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.axvline(x=x_tick_positions[x_tick_labels == 0][0], c='k', linestyle='--')
    ax.set_xlabel('Binned Time (s)')

    y_tick_labels = np.arange(1, traces.shape[0]+1, 5)
    y_tick_positions = np.linspace(0, traces.shape[0]-1, len(y_tick_labels))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Episode')

    ax.set_title('PETH - {}'.format(scoring_type))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label=f_trace)

    plt.tight_layout()
    plt_name = "peth_{}_zscore.png".format(scoring_type.lower().replace(' ', '_'))
    plt.savefig(join(paths.figure_directory, plt_name))

    return fig


if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregate_episodes.h5'
    aggregate_data_file = join(paths.processed_data_directory, aggregate_data_filename)

    # -- Which episode(s) do you want to look at?
    # If set to 'ALL', generates means for all episodes individually.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    #episodes_to_analyze = 'ALL'
    episodes_to_analyze = ['Eating Zone Minus']

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 0    # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = 10    # Seconds

    # -- The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # -- Length of the bins for the histgram
    bin_length = 0.5  # Seconds

    try:
        aggregate_store = pd.HDFStore(aggregate_data_file)
        aggregate_keys = aggregate_store.keys()
        print('The following episodes are available to analyze: {}'.format(aggregate_keys))

        if episodes_to_analyze is 'ALL':
            episodes_to_analyze = [ak.strip('/') for ak in aggregate_keys]

        for episode_name in episodes_to_analyze:
            # Pull the data for the type of episode
            all_episodes = aggregate_store.get(episode_name.lower().replace(' ', '_'))

            # -- Remove certain days/animals
            # episodes_to_run = all_episodes.loc[all_episodes["day"] == 3]    # select day 3 exps
            # episodes_to_run = all_episodes.loc[all_episodes["animal"] != 1]    # remove animal 1
            # only day 3 experiments excluding animal 1
            # episodes_to_run = all_episodes.loc[(all_episodes["animal"] != 1) & (all_episodes["day"] == 3)]
            episodes_to_run = all_episodes

            # Do filtering. The function names are self-explanatory. If a value error is thrown,
            # that means the filtering removed all the episodes from the behaviors, and
            # that you need to change the filtering parameters for that kind of behavior
            try:
                episodes_to_run = f_aggr.filter_episodes_for_overlap(episodes_to_run)
                episodes_to_run = f_aggr.filter_episodes_by_duration(episodes_to_run, episode_duration_cutoff)
                episodes_to_run = f_aggr.filter_first_n_episodes(episodes_to_run, -1)
            except ValueError as e:
                print(e)
                print('Error in filtering parameters for {}! Change the parameters and re-run.'.format(episode_name))
                print('Moving to next episode type...\n')
                continue

            # Select the amount of time after the onset of an episode to look at
            episodes_to_run = f_aggr.select_analysis_window(episodes_to_run, post_onset_window)

            # Plot peri-event time histogram for the individual behaviors
            # (as selected in 'episodes_to_analyze') If you wanted to plot for a
            # DC experiment, it would look like
            # plot_peth(episodes_to_run, bin_length, episode_name, f_trace='zscore', channel_key='anterior'))
            plot_peth(episodes_to_run, bin_length, episode_name)

            # plt.show()

        aggregate_store.close()

    except FileNotFoundError as e:
        print(e)

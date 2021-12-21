import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

import paths
import fp.aggregation as f_aggr
from fp.utils import list_lists_to_array, remove_baseline, check_if_dual_channel_recording
from fp.visualization import fluorescence_axis_labels


def plot_trace_raster(episodes, scoring_type,
                      f_trace='zscore_Lerner', channel_key=None,
                      index_key='overall_episode_number', **kwargs):

    """ Plots a peri-event time histogram of individual episodes of some behavior.

    Parameters
    ----------
    episodes : pd.DataFrame
        pd.DataFrames containing fluorescence data for all episodes of a scoring types
    scoring_type: str
        Name of the episodes being plotting
    f_trace : str, default='zscore_Lerner'
        The fluorescence trace to be plotted.
        Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner].
    channel_key : str, optional, default=None
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default=None for single-fiber recordings.
    index_key : str, default='overall_episode_number'
        Name of the column in 'episodes' used for indexing

    Returns
    -------

    Keyword Arguments
    -----------------
    norm_start : float, int
        Time (normalized) at which trace baseline calculation starts
    norm_end : float, int
        Time (normalized) at which trace baseline calculation ends

    """

    # Handle keyword args. If these are not specified, the default to the values in parenthesis below.
    norm_start = kwargs.get('norm_start', -5)
    norm_end = kwargs.get('norm_end', 0)

    if channel_key is None:
        f_trace_key = f_trace
    else:
        f_trace_key = '_'.join([f_trace, channel_key])

    # Get episode traces
    times = episodes.groupby([index_key])['normalized_time'].agg(list).to_list()
    traces = episodes.groupby([index_key])[f_trace_key].agg(list).to_list()

    times = list_lists_to_array(times)
    traces = list_lists_to_array(traces)

    # The times should all be pretty similar (at least close enough for binning)
    # Take an average time trace for our calculation
    time = np.nanmean(times, axis=0)

    # Remove the baseline from the fluorescence traces in the window
    traces = remove_baseline(time, traces, norm_start=norm_start, norm_end=norm_end)

    # Create the figure
    fig, axes = plt.subplots(nrows=traces.shape[0], ncols=1, figsize=(10, 2*traces.shape[0]))

    for ax, trace in zip(axes, traces):
        ax.plot(time, trace)
        ax.set_ylim(np.nanmin(traces), np.nanmax(traces))

        ax.axvline(x=time[time == 0], c='k', linestyle='--')

        # Hide the right, top, and bottom spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Only show ticks on the left spine
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.yaxis.set_ticks_position('left')

    # Re-activate the bottom spine and ticks
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    # Label the axes
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_ylabel(fluorescence_axis_labels[f_trace])

    plt.suptitle('Raster traces for {} - {}'.format(scoring_type, f_trace))

    plt.tight_layout()
    plt_name = "raster_{}_{}.png".format(scoring_type.lower().replace(' ', '_'), f_trace)
    plt.savefig(join(paths.figure_directory, plt_name))

    return fig


if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregate_episodes.h5'
    aggregate_data_file = join(paths.preprocessed_data_directory, aggregate_data_filename)

    # -- Which episode(s) do you want to look at?
    # If set to 'ALL', generates means for all episodes individually.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    # episodes_to_analyze = 'ALL'
    episodes_to_analyze = ['Eating Zone Plus']

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 35    # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = 10    # Seconds

    # -- The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # -- Set the normalization window. This is the period where the baseline is calculated and subtracted from
    # the episode trace
    norm_start = -5
    norm_end = -2

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

            # Check if this is a dual-fiber experiment
            is_DC = check_if_dual_channel_recording(episodes_to_run)

            # Plot rastered fluorescence traces for the individual behaviors (as selected in 'episodes_to_analyze')
            if is_DC:
                channels = ['anterior', 'posterior']
                for channel in channels:
                    plot_trace_raster(episodes_to_run, episode_name,
                                      norm_start=norm_start, norm_end=norm_end,
                                      channel_key=channel)
            else:
                plot_trace_raster(episodes_to_run, episode_name, norm_start=norm_start, norm_end=norm_end)

            plt.show()

        aggregate_store.close()

    except FileNotFoundError as e:
        print(e)

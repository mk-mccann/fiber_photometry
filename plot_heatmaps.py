import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join
from scipy.stats import binned_statistic

import pandas as pd

import paths
import functions_aggregation as f_aggr
from functions_utils import list_lists_to_array, remove_baseline
from aggregate_episodes_across_experiments import create_episode_aggregate_h5


def plot_trace_heatmap(episodes, bin_duration, f_trace='zscore', index_key='overall_episode_number',
                       bin_function=np.nanmean, **kwargs):
    """

    Parameters
    ----------
    episodes
    bin_duration
    f_trace
    index_key
    bin_function

    Returns
    -------

    """

    # Get episode traces
    times = episodes.groupby([index_key])['normalized_time'].agg(list).to_list()
    traces = episodes.groupby([index_key])[f_trace].agg(list).to_list()

    times = list_lists_to_array(times)
    traces = list_lists_to_array(traces)

    # The times should all be pretty similar (at least close enough for binning)
    # Take an average time trace for our calculate
    time = np.nanmean(times, axis=0)

    # Remove the baseline from the fluorescence traces in the window
    traces = remove_baseline(time, traces, norm_start=-2)

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

    tick_labels = np.arange(bins[0], bins[-1]+bin_duration, 5)
    tick_positions = np.linspace(-0.5, len(bins)-1.5,  len(tick_labels))
    ax.axvline(x=tick_positions[tick_labels == 0][0], c='k', linestyle='--')

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Binned Time (s)')
    ax.set_ylabel('Episode')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label=f_trace)

    return fig




if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # create it and load it.
    aggregate_data_filename = 'aggregate_episodes.h5'
    aggregate_data_file = join(paths.processed_data_directory, aggregate_data_filename)

    episodes_to_analyze = ['Eating', 'Eating Zone Plus']
    episode_duration_cutoff = 35    # Seconds
    analysis_window = 10            # Seconds
    bin_length = 0.5                # Seconds

    try:
        aggregate_store = pd.HDFStore(aggregate_data_file)
        aggregate_keys = aggregate_store.keys()
        print('The following episodes are available to analyze: {}'.format(aggregate_keys))

        for episode in episodes_to_analyze:
            eta = episode.lower().replace(' ', '_')

            df = aggregate_store.get(eta)
            df = f_aggr.filter_episodes_for_overlap(df)
            df = f_aggr.filter_episodes_by_duration(df, episode_duration_cutoff)
            df = f_aggr.filter_first_n_episodes(df, -1)
            df = f_aggr.select_analysis_window(df, analysis_window)
            fig = plot_trace_heatmap(df, bin_length)

            fig.axes[0].set_title('PETH - {}'.format(episode))


            plt.tight_layout()
            plt_name = "peth_{}_zscore.png".format(eta)
            plt.savefig(join(paths.figure_directory, plt_name))

            # plt.show()

    except FileNotFoundError as e:
        print(e)

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join

import paths
import functions_aggregation as f_aggr
from functions_utils import check_if_dual_channel_recording
from plot_peri_event_time_histogram import plot_peth
from plot_episode_traces_raster import plot_trace_raster





if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregated_episodesPost_window.h5'
    aggregate_data_file = join(paths.preprocessed_data_directory, aggregate_data_filename)

    # -- Which episode(s) do you want to look at?
    # If set to 'ALL', generates means for all episodes individually.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    #episodes_to_analyze = 'ALL'
    episodes_to_analyze = ['Grooming']
    subset_to_plot = [1, 4, 5, 6, 8, 17]

    # Which fluorescence trace do you want to plot?
    # Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner]
    f_trace = 'zscore_Lerner'

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 0    # Seconds

    # -- How long after the onset of an episode do you want to look at?
    pre_onset_window = -5  # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = 7    # Seconds

    # -- The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # -- Length of the bins for the histogram
    bin_length = 0.2  # Seconds

    # -- Set the normalization window. This is the period where the baseline is calculated and subtracted from
    # the episode trace.
    norm_start = -3
    norm_end = 0

    try:
        aggregate_store = pd.HDFStore(aggregate_data_file)
        aggregate_keys = aggregate_store.keys()
        print('The following episodes are available to analyze: {}'.format(aggregate_keys))

        if episodes_to_analyze == 'ALL':
            episodes_to_analyze = [ak.strip('/') for ak in aggregate_keys]

        for episode_name in episodes_to_analyze:
            # Pull the data for the type of episode
            all_episodes = aggregate_store.get(episode_name.lower().replace(' ', '_'))

            # -- Remove certain days/animals
            # episodes_to_run = all_episodes.loc[all_episodes["day"] == 3]    # select day 3 exps
            #episodes_to_run = all_episodes.loc[all_episodes["animal"] != "2"]    # remove animal 1
            # only day 3 experiments excluding animal 1
            #episodes_to_run = all_episodes.loc[(all_episodes["animal"] != 1) & (all_episodes["day"] == 3)]
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
            episodes_to_run = f_aggr.select_analysis_window(episodes_to_run, pre_onset_window, post_onset_window)

            # select specific episodes from a list
            episode_subset = episodes_to_run[episodes_to_run.overall_episode_number.isin(subset_to_plot)]

            # Check if this is a dual-fiber experiment
            is_DC = check_if_dual_channel_recording(episode_subset)

            # Plot peri-event time histogram for the individual behaviors (as selected in 'episodes_to_analyze')
            if is_DC:
                channels = ['anterior', 'posterior']
                for channel in channels:
                    _, peth_ax = plot_peth(episode_subset, bin_length, episode_name,
                                                  norm_start=norm_start, norm_end=norm_end,
                                                  channel_key=channel, cmap_lim=4)

                    _, raster_axes = plot_trace_raster(episode_subset, episode_name,
                                                                norm_start=norm_start, norm_end=norm_end,
                                                                channel_key=channel)
            else:

                num_rasters = len(subset_to_plot)

                fig = plt.figure(figsize=(15, 9))
                subfigs = fig.subfigures(1, 2, width_ratios=[0.5, 0.75])
                axsLeft = subfigs[0].subplots(num_rasters, 1, sharex=True, sharey=True,
                                              gridspec_kw={'height_ratios': [0.1 for i in range(num_rasters)]})
                _ = plot_trace_raster(episode_subset, episode_name,
                                      norm_start=norm_start, norm_end=norm_end, axes=axsLeft)
                subfigs[0].text(0.04, 0.5, 'Z dF/F', va='center', rotation='vertical')

                axsRight = subfigs[1].subplots(1, 1)
                peth_ax, peth_im = plot_peth(episode_subset, bin_length, episode_name,
                              norm_start=norm_start, norm_end=norm_end, cmap_lim=2, ax=axsRight)
                subfigs[1].colorbar(peth_im, shrink=0.4, ax=axsRight, location='bottom', label='z-score')

                plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.7, wspace=0, hspace=0.1)

            plt.show()

        aggregate_store.close()

    except FileNotFoundError as e:
        print(e)

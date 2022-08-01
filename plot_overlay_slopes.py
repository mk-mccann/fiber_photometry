import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
from os.path import join

import paths
import functions_aggregation as f_aggr
from plot_mean_episode import plot_mean_episode
from get_slopes import get_slopes
import functions_plotting as fp
from functions_utils import find_nearest, check_if_dual_channel_recording, list_lists_to_array, remove_baseline


if __name__ == "__main__":

    # Path to wild_type experiment:
    wt_path = r"E:\Alja_FP\January '21 WILD TYPE\Multimaze\preprocessed_data"

    # Path to dual channel experiment
    dc_path = r"E:\Alja_FP\July '21 DUAL CHANNEL\Multimaze\preprocessed_data"

    # Episode
    episode_name = 'Eating Window'

    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregated_20_sec.h5'
    wt_aggregate_data_file = join(wt_path, aggregate_data_filename)
    dc_aggregate_data_file = join(dc_path, aggregate_data_filename)

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 0    # Seconds

    # -- How long before the onset of an episode do you want to look at?
    pre_onset_window = -3  # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = 8    # Seconds

    # -- The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # -- Set the normalization window. This is the period where the baseline is calculated and subtracted from
    # the episode trace
    norm_start = -3
    norm_end = -0

    try:
        wt_aggregate_store = pd.HDFStore(wt_aggregate_data_file)
        dc_aggregate_store = pd.HDFStore(dc_aggregate_data_file)

        wt_all_episodes = wt_aggregate_store.get(episode_name.lower().replace(' ', '_'))
        dc_all_episodes = dc_aggregate_store.get(episode_name.lower().replace(' ', '_'))

        wt_aggregate_store.close()
        dc_aggregate_store.close()

        # -- Remove certain days/animals
        #episodes_to_run = all_episodes.loc[all_episodes["day"] == "3"]    # select day 3 exps
        #episodes_to_run = all_episodes.loc[all_episodes["animal"] != "1"]    # remove animal 1
        # only day 3 experiments excluding animal 1
        #episodes_to_run = all_episodes.loc[(all_episodes["animal"] == "5") & (all_episodes["day"] == "2")]
        #episodes_to_run = all_episodes.loc[(all_episodes["zscore_Lerner"] <= 2.0) & (all_episodes["zscore_Lerner"] >= -2.0)]
        wt_episodes_to_run = wt_all_episodes
        dc_episodes_to_run = dc_all_episodes

        # Do filtering. The function names are self-explanatory. If a value error is thrown,
        # that means the filtering removed all the episodes from the behaviors, and
        # that you need to change the filtering parameters for that kind of behavior
        try:
            # wt_episodes_to_run = f_aggr.filter_episodes_for_overlap(wt_episodes_to_run)
            wt_episodes_to_run = f_aggr.filter_episodes_by_duration(wt_episodes_to_run, episode_duration_cutoff)
            wt_episodes_to_run = f_aggr.filter_first_n_episodes(wt_episodes_to_run, first_n_eps)

            # dc_episodes_to_run = f_aggr.filter_episodes_for_overlap(dc_episodes_to_run)
            dc_episodes_to_run = f_aggr.filter_episodes_by_duration(dc_episodes_to_run, episode_duration_cutoff)
            dc_episodes_to_run = f_aggr.filter_first_n_episodes(dc_episodes_to_run, first_n_eps)

        except ValueError as e:
            print(e)
            print('Error in filtering parameters for {}! Change the parameters and re-run.'.format(episode_name))
            print('Moving to next episode type...\n')
            pass

        # Select the amount of time after the onset of an episode to look at
        wt_episodes_to_run = f_aggr.select_analysis_window(wt_episodes_to_run, pre_onset_window, post_onset_window)
        dc_episodes_to_run = f_aggr.select_analysis_window(dc_episodes_to_run, pre_onset_window, post_onset_window)

        # Plot means for the individual behaviors (as selected in 'episodes_to_analyze')
        # If you wanted to plot for a DC experiment, it would look like
        # plot_individual_behaviors(episodes_to_run, f_trace='zscore', channel_key='anterior))
        fig_wt = get_slopes(wt_episodes_to_run, episode_name,
                   plot_singles=False, norm_start=norm_start, norm_end=norm_end,)

        fig_overlay = get_slopes(dc_episodes_to_run, episode_name,
                              plot_singles=False, norm_start=norm_start, norm_end=norm_end,
                              channel_key='anterior', ax=fig_wt, fill_between_color='Eating Window DC')

        fig_overlay.set_ylim(-0.5, 2.5)

        plt.show()

    except FileNotFoundError as e:
        print(e)

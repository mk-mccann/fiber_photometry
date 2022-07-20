import numpy as np
import pandas as pd
from os.path import join

import paths
import functions_aggregation as f_aggr
from functions_utils import list_lists_to_array, remove_baseline, check_if_dual_channel_recording, find_nearest


def find_peaks(episodes, scoring_type, f_trace='zscore_Lerner', channel_key=None, index_key='overall_episode_number', **kwargs):

    # Handle keyword args. If these are not specified, the default to the values in parentheses below.
    norm_start = kwargs.get('norm_start', -5)
    norm_end = kwargs.get('norm_end', 0)

    if channel_key is None:
        f_trace_key = f_trace
    else:
        f_trace_key = '_'.join([f_trace, channel_key])

    # Get episode traces
    times = episodes.groupby([index_key])['normalized_time'].agg(list).to_list()
    episodes[f_trace_key].fillna(0, inplace=True)
    traces = episodes.groupby([index_key])[f_trace_key].agg(list)
    episode_idxs = traces.index.to_numpy()

    if scoring_type == 'Grooming':
        trace_peak_idx = traces.apply(np.nanargmin).to_numpy()
    else:
        trace_peak_idx = traces.apply(np.nanargmax).to_numpy()

    times = list_lists_to_array(times)
    traces = list_lists_to_array(traces)

    # The times should all be pretty similar (at least close enough for binning)
    # Take an average time trace for our calculation
    time = np.nanmean(times, axis=0)
    t_0_idx = np.argwhere(time == 0)[0][0]
    t_max_idx, _ = find_nearest(time, 15.0)

    # Remove the baseline from the fluorescence traces in the window
    traces = remove_baseline(time, traces, norm_start=norm_start, norm_end=norm_end)

    # Get extrema
    extrema = []
    extrema_time = []
    medians = []
    for trace, idx in zip(traces, trace_peak_idx):
        extrema.append(trace[idx])
        extrema_time.append(time[idx])

        if len(trace) > t_max_idx:
            medians.append(np.nanmedian(trace[t_0_idx:t_max_idx]))
        else:
            medians.append(np.nanmedian(trace[t_0_idx:]))

    # Get episode metadata
    episode_metadata = []
    cols = ['animal', 'day', 'exp_episode_number']
    for ep in episode_idxs:
        metadata = episodes[episodes[index_key] == ep][cols].drop_duplicates()
        episode_metadata.append(metadata.values.astype(float))

    # Get everything into a nice big dataframe so we can save it as a csv
    df_data_list = []
    df_columns = ['animal', 'day', 'behavior', 'exp_episode_number', 'overall_episode_number', 'peak', 'time_to_peak', 'median']
    for i, ep_num in enumerate(episode_idxs):
        animal, day, exp_ep_num = episode_metadata[i].tolist()[0]
        overall_ep_num = ep_num
        ep_peak = extrema[i]
        ep_peak_time = extrema_time[i]
        ep_median = medians[i]
        ep_data = [animal, day, scoring_type, exp_ep_num, overall_ep_num, ep_peak, ep_peak_time, ep_median]
        df_data_list.append(ep_data)

    df = pd.DataFrame(data=df_data_list, columns=df_columns)

    if channel_key is not None:
        df['channel'] = channel_key

    return df


if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregated_episodes_no_post_window.h5'
    aggregate_data_file = join(paths.preprocessed_data_directory, aggregate_data_filename)

    # -- Which episode(s) do you want to look at?
    # If set to 'ALL', generates means for all episodes individually.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    # episodes_to_analyze = 'ALL'
    episodes_to_analyze = ['Grooming', 'Eating Window', 'Shock', 'Social_Interaction', 'Transfer']

    # Give a subset of trials to consider. If you want to plot them all, leave the list empty []
    subset_to_plot = []

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 0    # Seconds

    # -- How long before the onset of an episode do you want to look at?
    pre_onset_window = -3  # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = -1    # Seconds

    # -- The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # -- Set the normalization window. This is the period where the baseline is calculated and subtracted from
    # the episode trace
    norm_start = -3
    norm_end = 0

    df_list = []
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
            # episodes_to_run = all_episodes.loc[all_episodes["animal"] != 1]    # remove animal 1
            # only day 3 experiments excluding animal 1
            # episodes_to_run = all_episodes.loc[(all_episodes["animal"] != 1) & (all_episodes["day"] == 3)]
            episodes_to_run = all_episodes

            # Do filtering. The function names are self-explanatory. If a value error is thrown,
            # that means the filtering removed all the episodes from the behaviors, and
            # that you need to change the filtering parameters for that kind of behavior
            try:
                episodes_to_run = f_aggr.filter_episodes_for_overlap(episodes_to_run)
                episodes_to_run = f_aggr.filter_episodes_by_duration(episodes_to_run, episode_duration_cutoff, filter_type='greater_than')
                episodes_to_run = f_aggr.filter_first_n_episodes(episodes_to_run, first_n_eps)
            except ValueError as e:
                print(e)
                print('Error in filtering parameters for {}! Change the parameters and re-run.'.format(episode_name))
                print('Moving to next episode type...\n')
                continue

            # Select the amount of time after the onset of an episode to look at
            episodes_to_run = f_aggr.select_analysis_window(episodes_to_run, pre_onset_window, post_onset_window)

            # select specific episodes from a list
            if len(subset_to_plot) > 0:
                episodes_to_run = episodes_to_run[episodes_to_run.overall_episode_number.isin(subset_to_plot)]

            # Check if this is a dual-fiber experiment
            is_DC = check_if_dual_channel_recording(episodes_to_run)

            # Plot raster fluorescence traces for the individual behaviors (as selected in 'episodes_to_analyze')
            if is_DC:
                channels = ['anterior', 'posterior']
                for channel in channels:
                    peak_df = find_peaks(episodes_to_run, episode_name,
                                      norm_start=norm_start, norm_end=norm_end,
                                      channel_key=channel)
            else:
                peak_df = find_peaks(episodes_to_run, episode_name, norm_start=norm_start, norm_end=norm_end)

            df_list.append(peak_df)

        aggregate_store.close()

        peaks = pd.concat(df_list)

        # A bit of stupidity to make sure that the animal names are consistent
        peaks['animal'][peaks['animal'] == 3.2] = 3.0
        peaks.animal = peaks.animal.astype(int)
        peaks.day = peaks.day.astype(int)
        peaks.exp_episode_number = peaks.exp_episode_number.astype(int)

        savename = f"peaks_{'_'.join(episodes_to_analyze).lower()}.csv"
        peaks.to_csv(join(paths.preprocessed_data_directory, savename), index=False)
        print('done!')

    except FileNotFoundError as e:
        print(e)

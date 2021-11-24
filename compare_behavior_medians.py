import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from os.path import join

import paths
import functions_utils as f_util
import functions_aggregation as f_aggr


def median_comparison(episodes, f_trace='zscore_Lerner', channel_key=None, plot_singles=False):
    """Creates and saves a plot of the mean fluorescence trace across all episodes of the individual scoring types
    contained in the input 'data_dict'. Plots mean + SEM.

    Parameters
    ----------
    episodes : pd.DataFrame
        pd.DataFrames containing fluorescence data for all episodes of a scoring types
    f_trace : str, default='zscore_Lerner'
        The fluorescence trace to be plotted.
        Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner].
    channel_key : str, optional
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default is None for single-fiber recordings.
    plot_singles : bool, default=False
        Boolean value to plot individual episode traces.

    Returns
    -------

    See Also
    --------
    plot_mean_episode : Plots mean + SEM of all input traces

    """

    if channel_key is None:
        f_trace = f_trace
    else:
        f_trace = '_'.join([f_trace, channel_key])

    median_dict = {}    

    # Loop through all the conditions we pulled out before and plot them
    for k in data_dict.keys():
        f_traces_of_key = [df[f_trace].to_numpy() for df in data_dict[k]]

        if len(f_traces_of_key) > 0:
            k_dict = {}
            
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} episodes = {}".format(k, num_episodes))

            ep_medians = np.nanmedian(trace_array, axis=1)
            ep_mean = np.mean(ep_medians)
            ep_sem = sem(ep_medians)
            
            median_dict[k] = ep_medians

        else:
            print("No episodes of {} found!".format(k))
            
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in median_dict.items()]))


if __name__ == "__main__":

    # Check if an aggregated episode file exists. If so, load it. If not,
    # throw an error
    aggregate_data_filename = 'aggregate_episodes.h5'
    aggregate_data_file = join(paths.preprocessed_data_directory, aggregate_data_filename)

    # -- Which episode(s) do you want to look at?
    # If set to 'ALL', generates means for all episodes individually.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    episodes_to_analyze = 'ALL'  # ['Eating', 'Eating Zone Plus']

    # -- What is the amount of time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    episode_duration_cutoff = 35  # Seconds

    # -- How long after the onset of an episode do you want to look at?
    post_onset_window = 10  # Seconds

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

            # Plot means for the individual behaviors (as selected in 'episodes_to_analyze')
            # If you wanted to plot for a DC experiment, it would look like
            # plot_individual_behaviors(episodes_to_run, f_trace='zscore', channel_key='anterior))
            median_comparison(episodes_to_run, episode_name,
                              plot_singles=True, norm_start=norm_start, norm_end=norm_end
                              )

            plt.show()

        aggregate_store.close()

    except FileNotFoundError as e:
        print(e)



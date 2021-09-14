import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import paths
import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors, plot_mean_episode


def get_individual_episode_indices(data_df, key):
    """Finds the DataFrame indices of all episodes of a given scoring type, and breaks them into individual episodes

    Parameters
    ----------
    data_df : pd.DataFrame
        Aggregated data from all experiments
    key : str
        Scoring type to be searched for

    Returns
    -------
    valid_episode_idxs : list of np.arrays
        Indices of valid episodes. If no episodes of a given scoring type, returns an empty list.

    """

    valid_episode_idxs = []

    # Get the starting and ending indexes and times of all episodes of a given type
    if 'Zone' in key:
        # Here is a special case for the eating zone. We only want to look at times in the eating zone when the mouse
        # is actually eating
        if ('Eating' in key) and ('+' in key):
            episode_idxs = (data_df[(data_df['zone'] == 'Eating Zone') & (data_df['behavior'] == 'Eating')].index.to_numpy())
        elif ('Eating' in key) and ('-' in key):
            episode_idxs = (data_df[(data_df['zone'] == 'Eating Zone') & (data_df['behavior'] != 'Eating')].index.to_numpy())
        else:
            episode_idxs = data_df[data_df['zone'] == key].index.to_numpy()
    else:
        episode_idxs = data_df[data_df['behavior'] == key].index.to_numpy()

    # episodes_to_plot is a list of all indices fulfilling the behavior or zone occupancy condition.
    # We want specific episodes, so split the list. If there are no episodes of a given behavior, return
    # an empty list
    if episode_idxs.size > 0:
        breaks = np.where(np.diff(episode_idxs) != 1)[0] + 1  # add 1 to compensate for the diff
        valid_episode_idxs = np.array_split(episode_idxs, breaks)
    else:
        pass

    return valid_episode_idxs


def check_episode_duration(data_df, episodes, min_dwell_time=0):
    """Checks the duration of an episode against a minimum duration to filter out short episodes.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of aggregated experiments
    episodes : iterable object
        Episode indices for a scoring type
    min_dwell_time : int, default=0
        Minimum duration of an episode in seconds for it to be considered valid. Default is 0 (keep all episodes).

    Returns
    -------
    valid_episodes : list of np.arrays
        Episodes meeting the dwell time requirement. If there are none, return an empty list.

    """

    valid_episodes = []

    for ep in episodes:
        start_idx = ep[0]
        end_idx = ep[-1]
        start_time = data_df['time'].iloc[start_idx]
        end_time = data_df['time'].iloc[end_idx]
        duration = end_time - start_time

        # Dwell filter: animal must spend at least this much time doing a behavior
        if duration >= min_dwell_time:
            valid_episodes.append(ep)

    return valid_episodes


def get_episode_start_window(data_df, episodes, window_period=(-5, 5)):
    """Checks the duration of an episode against a minimum duration to filter short scoring type epochs.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of aggregated experiments
    episodes : iterable object
        Episode indices for a given scoring type
    window_period : tuple or list of ints, default=(-5, 5)
        Duration in seconds to select before and after the onset of a scoring type.

    Returns
    -------
    start_windows : list of pd.DataFrames
        List of windows surrounding the start of a scoring type epoch of interest

    """

    start_windows = []

    # Some variables to number the episodes of a behavior within a given experiment
    ep_number = 1
    last_animal = 0
    last_day = 0

    for ep in episodes:
        ep_start_idx = ep[0]

        # For getting the start windows, we need to look at the experiment that a given episode occurred in.
        animal = data_df['animal'].iloc[ep_start_idx]
        day = data_df['day'].iloc[ep_start_idx]
        exp = data_df.loc[(data_df['animal'] == animal) & (data_df['day'] == day)]

        # Check to see if this animal or day changed from the last episode in the list. If so, reset counter.
        # Otherwise advance the counter
        if (float(animal) != last_animal) or (int(day) != last_day):
            ep_number = 1
        else:
            ep_number += 1

        ep_start_time = exp['time'][exp.index == ep_start_idx].item()

        _, window_start_time = f_util.find_nearest(exp['time'], ep_start_time + window_period[0])
        _, window_end_time = f_util.find_nearest(exp['time'], ep_start_time + window_period[1])

        try:
            start_idx = exp.loc[exp['time'] == window_start_time].index.item()
        except IndexError:
            start_idx = exp.loc[exp['time'] == exp['time'].min()].index.item()

        try:
            end_idx = exp.loc[exp['time'] == window_end_time].index.item()
        except IndexError:
            end_idx = exp.loc[exp['time'] == exp['time'].max()].index.item()

        ep_df = data_df.iloc[start_idx:end_idx].copy()
        ep_df['episode_number'] = ep_number
        start_windows.append(ep_df)

        last_animal = float(animal)
        last_day = int(day)

    return start_windows


def take_first_n_episodes(episodes, n_to_keep=-1):
    """Select the first N episodes of a given scoring types in each experiment.

    Parameters
    ----------
    episodes : list of np.arrays
        Episode indices for a scoring type
    n_to_keep: int, optional
        First N episodes to keep. Default is -1 (keep all episodes).

    Returns
    -------
    episodes_to_keep : list of pd.DataFrames
        First N episodes of a desired scoring type in each experiment
    """

    episodes_to_keep = []

    if n_to_keep == -1:
        episodes_to_keep = episodes
    else:
        for ep in episodes:
            episode_number = int(ep['episode_number'].iloc[0])

            if episode_number <= n_to_keep:
                episodes_to_keep.append(ep)

    return episodes_to_keep


def plot_individual_behaviors(data_dict, f_trace='zscore', channel_key=None, plot_singles=False):
    """Creates and saves a plot of the mean fluorescence trace across all episodes of the individual scoring types
    contained in the input 'data_dict'. Plots mean + SEM.

    Parameters
    ----------
    data_dict : dict
        Dictionary where each key is a list of pd.DataFrames containing fluorescence data for all episodes of that
        scoring types
    f_trace : str, default='zscore'
        The fluorescence trace to be plotted. Options are ['auto', 'gcamp', 'dff', 'zscore'].
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

    # Loop through all the conditions we pulled out before and plot them
    for k in data_dict.keys():
        f_traces_of_key = [df[f_trace].to_numpy() for df in data_dict[k]]

        if len(f_traces_of_key) > 0:
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} episodes = {}".format(k, num_episodes))

            t = np.linspace(period[0], period[1], trace_array.shape[-1])

            # Remove the baseline using the 5 seconds before behavior onset
            trace_array = f_util.remove_baseline(t, trace_array)

            # Plot the mean episode
            fig = plot_mean_episode(t, trace_array, plot_singles=plot_singles)
            plt.ylabel('Z-dF/F')
            plt.title('Mean trace for {}'.format(k))
            plt_name = "mean_{}_dff_zscore.png".format(k.lower().replace(' ', '_'))
            plt.savefig(join(paths.figure_directory, plt_name))
        else:
            print("No episodes of {} found!".format(k))


def plot_multiple_behaviors(data_dict, keys_to_plot, f_trace='zscore', channel_key=None, plot_singles=False):
    """Creates and saves a plot of the mean fluorescence trace across all episodes of multiple scoring types contained
    in the input data_dict. Plots mean + SEM.

    Parameters
    ----------
    data_dict : dict
        Dictionary where each key is a list of pd.DataFrames containing fluorescence data for all episodes of that
        scoring types
    keys_to_plot: list of str
       The scoring types to be plotted.
    f_trace : str, default='zscore'
        The fluorescence trace to be plotted. Options are ['auto', 'gcamp', 'dff', 'zscore'].
    channel_key : str, optional
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default is None for single-fiber recordings.
    plot_singles : bool, default False
        Boolean value to plot individual episode traces.

    Returns
    -------

    See Also
    --------
    plot_mean_episode : Plots mean + SEM of all input traces
    """

    collected_traces = []

    if channel_key is None:
        f_trace = f_trace
    else:
        f_trace = '_'.join([f_trace, channel_key])

    for k in keys_to_plot:
        f_traces_of_key = [df[f_trace].to_numpy() for df in data_dict[k]]

        if len(f_traces_of_key) > 0:
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} episodes = {}".format(k, num_episodes))

            t = np.linspace(period[0], period[1], trace_array.shape[-1])

            # Remove the baseline using the 5 seconds before behavior onset
            # trace_array = f_util.remove_baseline(t, trace_array, norm_start=-5, norm_end=0)

            collected_traces.append(trace_array)

        else:
            print("No episodes of {} found!".format(k))

    collected_traces = np.vstack(collected_traces)
    # Plot the mean episode
    fig = plot_mean_episode(t, collected_traces, plot_singles=plot_singles)
    plt.ylabel('Z-dF/F')
    plt.title('Mean trace for {}'.format(', '.join(keys_to_plot)))
    plt_name = "mean_{}_dff_zscore.png".format('_'.join(keys_to_plot))
    plt.savefig(join(paths.figure_directory, plt_name))
    return fig


def extract_episodes(data_df, analysis_period, types_to_analyze, min_dwell_time=0, extract_first_n=-1):
    """Pulls the individual episodes of a given scoring type(s) from the data frame of aggregated experiments.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data frame of the aggregated experiments
    analysis_period : tuple of ints
        The time in seconds before and after onset of a a given scoring type to analyze
    types_to_analyze: str or list of str
        The scoring types to be analyzed. If set to "ALL", analyzes all behaviors as given in the 'episode_colors'
        dictionary
    min_dwell_time : int, optional
        Minimum duration of a scoring type in seconds for it to be considered valid. Default is 0.
    extract_first_n : int, optional
        First N episodes to keep. Default is -1 (keep all episodes).

    Returns
    -------
    output_dict : dict
        Keys are the scoring types, and the contain list of pd.DataFrames for individual episodes

    """

    # Create a dictionary to hold the output traces
    if types_to_analyze == 'ALL':
        output_dict = {e: [] for e in episode_colors.keys()}
    else:
        output_dict = {e: [] for e in types_to_analyze}

    # Build a special case to handle the Eating Zone: Animal eating vs animal not eating
    if 'Eating Zone' in output_dict.keys():
        output_dict['Eating Zone +'] = []
        output_dict['Eating Zone -'] = []
        types_to_analyze = output_dict.keys()

    # Loop through each of the scoring types you are interested in analyzing in order to extract them.
    for scoring_type in output_dict.keys():
        # Finds individual episodes of a scoring type
        episodes_by_idx = get_individual_episode_indices(data_df, scoring_type)

        # This is a check if there are episodes for a given scoring type. If there are, it adds them to the output dict
        if len(episodes_by_idx) > 0:
            # Check the duration of each episode, and throw out any that are too short. Default is to keep everything.
            episodes_by_duration = check_episode_duration(data_df, episodes_by_idx, min_dwell_time=min_dwell_time)
            # Extracts a window surrounding the start of an episode, as given by the variable window_period
            ep_start_windows = get_episode_start_window(data_df, episodes_by_duration, window_period=analysis_period)
            # Take first n episodes of a behavior from each experiment. Default is to keep all (-1)
            first_n_eps = take_first_n_episodes(ep_start_windows, n_to_keep=extract_first_n)
            output_dict[scoring_type] = first_n_eps
        else:
            print('No episodes of {} found!'.format(scoring_type))

    return output_dict


if __name__ == "__main__":

    # Check if the figure-saving directory exists
    f_io.check_dir_exists(paths.figure_directory)    

    # Read the summary file as a giant pandas dataframe
    all_exps = f_io.load_all_experiments()

    # remove certain days/animals
    # exps_to_run = all_exps.loc[all_exps["day"] == 3]    # select day 3 exps
    # exps_to_run = all_exps.loc[all_exps["animal"] != 1]    # remove animal 1
    exps_to_run = all_exps

    # Which behavior(s) do you want to look at?
    # If set to 'ALL', generates means for all behaviors.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    behaviors_to_analyze = ['Eating', 'Eating Zone']
    period = (-10, 10)    # In seconds
    
    # What is the minimum time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    dwell_time = 0    # In seconds

    # The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1

    # Run the main function
    all_episodes = extract_episodes(exps_to_run, period, behaviors_to_analyze,
                                    min_dwell_time=dwell_time, extract_first_n=first_n_eps)


    # Plot means for the individual behaviors (as selected in behaviors_to_analyze)
    # If you wanted to plot for a DC experiment, it would look like
    # plot_individual_behaviors(all_episodes, f_trace='zscore', channel_key='anterior))
    plot_individual_behaviors(all_episodes)

    # Plot means across all or some of the behaviors
    # If you set key = 'ALL' initially, and want to just look at a subset of behaviors,
    # then you need change multi_behav_plot with a list of the behaviors you want
    # to see. The example below plots all behaviors, but no zone occupancies
    multi_behav_plot = [key for key in list(all_episodes.keys()) if 'Zone' not in key]

    # Another example could be for only social interaction episodes and when the animal is eating in the eating zone
    # multi_behav_plot = ['Social Interaction Zone', 'Eating Zone +']

    # plot_multiple_behaviors(all_episodes, multi_behav_plot)

    # If you wanted to plot for a DC experiment, it would look like
    # plot_multiple_behaviors(all_episodes, multi_behav_plot, f_trace='zscore', channel_key='anterior)
    
    plt.show()

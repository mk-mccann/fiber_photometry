import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from os.path import join

import paths
import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors, plot_mean_episode
from plot_mean_behavior import get_individual_episode_indices, check_episode_duration, take_first_n_episodes


def get_episode(data_df, episodes):
    """Checks the duration of an episode against a minimum duration to filter short scoring type epochs.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of aggregated experiments
    episodes : iterable object
        Episode indices for a given scoring type
    window_period : tuple or list of ints, default=(0, -1)
        Duration in seconds to select before and after the onset of a scoring type. If the second value is -1, then the whole episode is considered

    Returns
    -------
    start_windows : list of pd.DataFrames
        List of windows surrounding the start of a scoring type epoch of interest

    """

    windows = []

    # Some variables to number the episodes of a behavior within a given experiment
    ep_number = 1
    last_animal = 0
    last_day = 0

    for ep in episodes:

        # For getting the start windows, we need to look at the experiment that a given episode occurred in.
        animal = data_df['animal'].iloc[ep[0]]
        day = data_df['day'].iloc[ep[0]]
        exp = data_df.loc[(data_df['animal'] == animal) & (data_df['day'] == day)]
        
        # Check to see if this animal or day changed from the last episode in the list. If so, reset counter.
        # Otherwise advance the counter
        if (float(animal) != last_animal) or (int(day) != last_day):
            ep_number = 1
        else:
            ep_number += 1

        ep_df = data_df.iloc[ep].copy()
        ep_df['episode_number'] = ep_number
        windows.append(ep_df)

        last_animal = float(animal)
        last_day = int(day)

    return windows



def extract_episodes(data_df, types_to_analyze, min_dwell_time=0, extract_first_n=-1):
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
            ep_start_windows = get_episode(data_df, episodes_by_duration)
            # Take first n episodes of a behavior from each experiment. Default is to keep all (-1)
            first_n_eps = take_first_n_episodes(ep_start_windows, n_to_keep=extract_first_n)
            output_dict[scoring_type] = first_n_eps
        else:
            print('No episodes of {} found!'.format(scoring_type))

    return output_dict


def plot_median_comparison(data_dict, f_trace='zscore', channel_key=None, plot_singles=False):
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
            
    return pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in median_dict.items() ]))


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
    
    # What is the minimum time an animal needs to spend performing a behavior or
    # being in a zone for it to be considered valid?
    dwell_time = 0    # In seconds

    # The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use first_n_eps = 2
    first_n_eps = -1
    
    # Run the main function
    all_episodes = extract_episodes(exps_to_run, behaviors_to_analyze,
                                    min_dwell_time=dwell_time, extract_first_n=first_n_eps)
    
    a = plot_median_comparison(all_episodes)
    a.to_csv(join(paths.fp_data_root_dir, 'median_csv.csv'))

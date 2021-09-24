import numpy as np
import pandas as pd

import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors


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


def get_episode_with_start_window(data_df, episodes, start_window=-5):
    """Checks the duration of an episode against a minimum duration to filter short scoring type epochs.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of aggregated experiments
    episodes : iterable object
        Episode indices for a given scoring type
    start_window : int, default=-5
        Duration in seconds to select before the onset of a scoring type.

    Returns
    -------
    start_windows : list of pd.DataFrames
        List of windows surrounding the start of a scoring type epoch of interest

    """

    start_windows = []

    # Some variables to number the episodes of a behavior within a given experiment
    exp_ep_number = 1
    overall_ep_number = 1
    last_animal = 0
    last_day = 0

    for ep in episodes:
        ep_start_idx = ep[0]
        ep_end_idx = ep[-1]

        # For getting the start windows, we need to look at the experiment that a given episode occurred in.
        animal = data_df['animal'].iloc[ep_start_idx]
        day = data_df['day'].iloc[ep_start_idx]
        exp = data_df.loc[(data_df['animal'] == animal) & (data_df['day'] == day)]

        # Check to see if this animal or day changed from the last episode in the list. If so, reset counter.
        # Otherwise advance the counter
        if (float(animal) != last_animal) or (int(day) != last_day):
            exp_ep_number = 1
        else:
            exp_ep_number += 1

        ep_start_time = exp['time'][exp.index == ep_start_idx].item()

        _, window_start_time = f_util.find_nearest(exp['time'], ep_start_time + start_window)

        try:
            start_idx = exp.loc[exp['time'] == window_start_time].index.item()
        except IndexError:
            start_idx = exp.loc[exp['time'] == exp['time'].min()].index.item()

        ep_df = data_df.iloc[start_idx:ep_end_idx].copy()
        ep_df['exp_episode_number'] = exp_ep_number
        ep_df['overall_episode_number'] = overall_ep_number
        start_windows.append(ep_df)

        last_animal = float(animal)
        last_day = int(day)

        overall_ep_number += 1

    return start_windows


def extract_episodes(data_df, analysis_period):
    """Pulls the individual episodes of a given scoring type(s) from the data frame of aggregated experiments.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data frame of the aggregated experiments
    analysis_period : int
        The time in seconds before onset of a given scoring type to analyze

    Returns
    -------
    output_dict : dict
        Keys are the scoring types, and the contain list of pd.DataFrames for individual episodes

    """

    # Create a dictionary to hold the output traces
    output_dict = {e: [] for e in episode_colors.keys()}

    # Build a special case to handle the Eating Zone: Animal eating vs animal not eating
    if 'Eating Zone' in output_dict.keys():
        output_dict['Eating Zone +'] = []
        output_dict['Eating Zone -'] = []

    # Loop through each of the scoring types you are interested in analyzing in order to extract them.
    for scoring_type in output_dict.keys():
        # Finds individual episodes of a scoring type
        episodes_by_idx = get_individual_episode_indices(data_df, scoring_type)

        # This is a check if there are episodes for a given scoring type. If there are, it adds them to the output dict
        if len(episodes_by_idx) > 0:
            # Extracts a window surrounding the start of an episode, as given by the variable 'start_window'
            ep_start_windows = get_episode_with_start_window(data_df, episodes_by_idx, start_window=analysis_period)
            scoring_type_df = pd.concat(ep_start_windows)
            output_dict[scoring_type] = scoring_type_df
        else:
            print('No episodes of {} found!'.format(scoring_type))

    return output_dict


if __name__ == '__main__':
    # Read the summary file as a giant pandas dataframe
    all_exps = f_io.load_all_experiments()

    # remove certain days/animals
    # exps_to_run = all_exps.loc[all_exps["day"] == 3]    # select day 3 exps
    # exps_to_run = all_exps.loc[all_exps["animal"] != 1]    # remove animal 1
    exps_to_run = all_exps

    # The period before/after the start of a scoring type
    period = -5

    # Run the main function
    all_episodes = extract_episodes(exps_to_run, period)

    # Save the dictionary to an .h5 file
    f_io.save_pandas_dict_to_h5(all_episodes, 'aggregated_behaviors.h5')


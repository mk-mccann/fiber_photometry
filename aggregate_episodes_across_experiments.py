import numpy as np
import pandas as pd

import functions_utils as f_util
import functions_io as f_io


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

    # Get the indices of all episodes of a given type
    if 'Zone' in key:
        # Here is a special case for the Eating Zone. We have three different Eating Zone types:
        # ('Eating Zone', 'Eating Zone Plus', and 'Eating Zone Minus') Just get the Eating Zone indices for all of them
        if 'Eating' in key:
            combined_episode_idxs = data_df[data_df['Eating Zone'] == 'Eating Zone'].index.to_numpy()
        else:
            combined_episode_idxs = data_df[data_df[key] == key].index.to_numpy()
    else:
        combined_episode_idxs = data_df[data_df[key] == key].index.to_numpy()

    # 'combined_episode_idxs' is a list of all indices fulfilling the behavior or zone occupancy condition.
    # We want specific episodes, so split the list. If there are no episodes of a given behavior, return
    # an empty list
    if combined_episode_idxs.size > 0:
        breaks = np.where(np.diff(combined_episode_idxs) != 1)[0] + 1  # add 1 to compensate for the diff
        split_episode_idxs = np.array_split(combined_episode_idxs, breaks)

        # Handle the 'Eating Zone Plus' and 'Eating Zone Minus' keys
        if key == 'Eating Zone Plus':
            for ep in split_episode_idxs:    
                ep_df_behav = data_df['Eating'].iloc[ep]
                if 'Eating' in ep_df_behav.unique():
                    valid_episode_idxs.append(ep)
        elif key == 'Eating Zone Minus':
            for ep in split_episode_idxs:    
                ep_df_behav = data_df['Eating'].iloc[ep]
                if 'Eating' not in ep_df_behav.unique():
                    valid_episode_idxs.append(ep)
        else:
            valid_episode_idxs = split_episode_idxs
        
    else:
        pass

    return valid_episode_idxs


def get_episode_with_start_window(data_df, episodes, pre_episode_window=-5):
    """Checks the duration of an episode against a minimum duration to filter short scoring type epochs.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame of aggregated experiments
    episodes : iterable object
        Episode indices for a given scoring type
    pre_episode_window : int, default=-5
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

        # Find the start time of the episode in the aggregated dataframe, and the beginning of the start window
        ep_start_time = exp['time'][exp.index == ep_start_idx].item()
        _, window_start_time = f_util.find_nearest(exp['time'], ep_start_time + pre_episode_window)

        # Extract the window of interest
        try:
            start_idx = exp.loc[exp['time'] == window_start_time].index.item()
        except IndexError:
            start_idx = exp.loc[exp['time'] == exp['time'].min()].index.item()

        ep_df = data_df.iloc[start_idx:ep_end_idx].copy()

        # Add some metadata to the episode dataframe
        ep_df['exp_episode_number'] = exp_ep_number
        ep_df['overall_episode_number'] = overall_ep_number
        ep_df['normalized_time'] = ep_df['time'] - ep_start_time

        start_windows.append(ep_df)

        last_animal = float(animal)
        last_day = int(day)
        overall_ep_number += 1

    return start_windows


def extract_episodes(data_df, pre_episode_window):
    """Pulls the individual episodes of a given scoring type(s) from the data frame of aggregated experiments.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data frame of the aggregated experiments
    pre_episode_window : int
        The time in seconds before onset of a given scoring type to analyze

    Returns
    -------
    output_dict : dict
        Keys are the scoring types, and the contain list of pd.DataFrames for individual episodes

    """

    # Create a dictionary to hold the output traces
    labeled_episodes = data_df.columns[data_df.columns.get_loc('zone') + 1:]
    output_dict = {e: [] for e in labeled_episodes}

    # Build a special case to handle the Eating Zone: Animal eating vs animal not eating
    if 'Eating Zone' in output_dict.keys():
        output_dict['Eating Zone Plus'] = []
        output_dict['Eating Zone Minus'] = []

    # Loop through each of the scoring types you are interested in analyzing in order to extract them.
    for scoring_type in output_dict.keys():
        # Finds individual episodes of a scoring type
        episodes_by_idx = get_individual_episode_indices(data_df, scoring_type)

        # This is a check if there are episodes for a given scoring type. If there are, it adds them to the output dict
        if len(episodes_by_idx) > 0:
            # Extracts a window surrounding the start of an episode, as given by the variable 'start_window'
            ep_start_windows = get_episode_with_start_window(data_df, episodes_by_idx, pre_episode_window=pre_episode_window)
            scoring_type_df = pd.concat(ep_start_windows)
            output_dict[scoring_type] = scoring_type_df
        else:
            print('No episodes of {} found!'.format(scoring_type))

    return output_dict


def create_episode_aggregate_h5(aggregate_df, pre_episode_window=-5, filename='aggregate_episodes.h5'):
    """Function

    Note that if there are no episodes of a scoring type in any experiment,
    they are note saved into the output file.

    Parameters
    ----------
    aggregate_df : pd.DataFrame
            Data frame of the aggregated experiments
    pre_episode_window : int
        The time in seconds before onset of a given scoring type to analyze
    filename : str, optional
        Name of the aggregated episode file to be saved

    """

    # Run the main function
    episodes = extract_episodes(aggregate_df, pre_episode_window)

    # Save the dictionary to an .h5 file
    if ('.h5' not in filename) or ('.h5py' not in filename):
        filename = filename + '.h5'

    f_io.save_pandas_dict_to_h5(episodes, filename)


if __name__ == '__main__':
    # Read all preprocessed data into a giant pandas dataframe
    # Note that here the behavior scoring must be done for an experiment to be
    # included in the aggregated data frame
    all_exps = f_io.load_all_experiments()

    # The period before the start of an episode
    period = -5

    # Run the main function
    all_episodes = extract_episodes(all_exps, period)

    # Save the dictionary to an .h5 file
    # Note that if there are no episodes of a scoring type in any experiment,
    # they are not saved into this file.
    f_io.save_pandas_dict_to_h5(all_episodes, 'aggregate_episodes.h5')
    print('Done!')

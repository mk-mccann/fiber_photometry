import numpy as np
import pandas as pd


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


def take_first_n_episodes(episodes, n_to_keep=-1):
    """DEPRECIATED: Select the first N episodes of a given scoring types in each experiment.

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


def filter_first_n_episodes(episodes, n_to_keep=-1):
    """Select the first N episodes of a given scoring types in each experiment.

        Parameters
        ----------
        episodes : pd.DataFrame
            DataFrame containing the aggregated episodes of the scoring type
        n_to_keep: int, optional
            First N episodes to keep. Default is -1 (keep all episodes).

        Returns
        -------
        episodes_to_keep : pd.DataFrame
            First N episodes of a desired scoring type in each experiment
        """

    if n_to_keep == -1:
        episodes_to_keep = episodes
        return episodes_to_keep
    elif n_to_keep == 0:
        print('Cannot keep 0 episodes!')
        raise IndexError
    else:
        episodes_to_keep = episodes[episodes['exp_episode_number'] <= n_to_keep]
        return episodes_to_keep


def sort_episodes_by_duration(episodes, key='overall_episode_number', sort_order='ascending'):

    time_traces = episodes.groupby(key)['time'].agg(list).to_list()
    durations = np.array([t[-1] - t[0] for t in time_traces])

    if sort_order is 'ascending':
        sort_idxs= np.argsort(durations)
    elif sort_order is 'descending':
        sort_idxs = np.argsort(durations)[::-1]
    else:
        raise AttributeError('This is not a valid sorting order!')

    sorted_durations = durations[sort_idxs]

    return sort_idxs, sorted_durations


def filter_episodes_by_duration(episodes, duration_cutoff, filter_type='greater_than', key='overall_episode_number'):
    sort_indxs, sort_durations = sort_episodes_by_duration(episodes, key)
    sort_ep_number = sort_indxs + 1    # Convert the index into episode number

    if filter_type is 'greater_than':
        good_episodes = sort_ep_number[np.argwhere(sort_durations >= duration_cutoff)].flatten()
    elif filter_type is 'less_than':
        good_episodes = sort_ep_number[np.argwhere(sort_durations <= duration_cutoff)].flatten()
    else:
        raise AttributeError('This is not a valid filter type!')

    valid_episodes = episodes[episodes[key].isin(good_episodes)]
    return valid_episodes


def filter_episodes_for_overlap(episodes, index_key='overall_episode_number'):
    good_episodes = []

    grouped_episodes = episodes.groupby([index_key])

    for ep, group in grouped_episodes:

        start_window_behavior = group[group['normalized_time'] < 0]['behavior'].unique()

        if (start_window_behavior.size == 1) and (start_window_behavior[0] is ''):
            good_episodes.append(ep)

    valid_episodes = episodes[episodes[index_key].isin(good_episodes)]

    return valid_episodes

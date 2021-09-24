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

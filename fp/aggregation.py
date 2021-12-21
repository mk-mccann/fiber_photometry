import pandas as pd


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
        valid_episodes : pd.DataFrame
            First N episodes of a desired scoring type in each experiment
        """

    if n_to_keep == -1:
        valid_episodes = episodes
        return valid_episodes
    elif n_to_keep == 0:
        print('Cannot keep 0 episodes!')
        raise IndexError
    else:
        valid_episodes = episodes[episodes['exp_episode_number'] <= n_to_keep]

    if valid_episodes.empty:
        raise ValueError
        # raise ValueError('The filtered DataFrame is empty! No episodes left to analyze!')
    else:
        return valid_episodes


def sort_episodes_by_duration(episodes, ascending=True, key='overall_episode_number'):

    sorted_episode_durations = episodes.groupby([key])['normalized_time'].max().sort_values(ascending=ascending)

    return sorted_episode_durations


def filter_episodes_by_duration(episodes, duration_cutoff, filter_type='greater_than', key='overall_episode_number'):
    sorted_durations = sort_episodes_by_duration(episodes, ascending=True, key=key)

    if filter_type == 'greater_than':
        good_episodes = sorted_durations[sorted_durations >= duration_cutoff].index.to_numpy()
    elif filter_type == 'less_than':
        good_episodes = sorted_durations[sorted_durations <= duration_cutoff].index.to_numpy()
    else:
        raise AttributeError('This is not a valid filter type!')

    valid_episodes = episodes[episodes[key].isin(good_episodes)]

    if valid_episodes.empty:
        raise ValueError
        # raise ValueError('The filtered DataFrame is empty! No episodes left to analyze!')
    else:
        return valid_episodes


def filter_episodes_for_overlap(episodes, index_key='overall_episode_number'):
    good_episodes = []

    grouped_episodes = episodes.groupby([index_key])

    for ep, group in grouped_episodes:

        start_window_behavior = group[group['normalized_time'] < 0]['behavior'].unique()

        if (start_window_behavior.size == 1) and (start_window_behavior[0] == ''):
            good_episodes.append(ep)

    valid_episodes = episodes[episodes[index_key].isin(good_episodes)]

    if valid_episodes.empty:
        raise ValueError
        # raise ValueError('The filtered DataFrame is empty! No episodes left to analyze!')
    else:
        return valid_episodes


def select_analysis_window(episodes, window, time_trace='normalized_time'):

    selection = episodes[episodes[time_trace] <= window]
    return selection



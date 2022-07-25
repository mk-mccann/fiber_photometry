import numpy as np
import pandas as pd
import math


def find_nearest(array: np.array, value: float):
    """ Find the closest-matching value to the input in the array.

    Parameters
    ----------
    array : np.array-like
        Array to search
    value : float
        Value to match

    Returns
    -------
    idx : int
        the index of the closest match in the array
    value : int or float
        The closest value
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def flatten_list(list_of_lists):
    """ Flattens a list of lists

    Parameters
    ----------
    list_of_lists : list

    Returns
    -------
        flattened list
    """

    return [item for sublist in list_of_lists for item in sublist]


def list_lists_to_array(list_of_lists):
    """ Converts a list of lists into a 2D array

    Parameters
    ----------
    list_of_lists : list

    Returns
    -------
    new_array : np.array
        Array where each row was an entry in the list of lists
    """

    max_length = max([len(sublist) for sublist in list_of_lists])
    new_array = np.empty((len(list_of_lists), max_length))
    new_array[:] = np.NaN

    for row, l in enumerate(list_of_lists):
        new_array[row, :len(l)] = l

    return new_array


def get_sec_from_min_sec(time: float):
    """ Converts a float representing time in min.sec to seconds

    Parameters
    ----------
    time : float

    Returns
    -------
    seconds : float
    """

    rounded_time = round(time, 2)
    split_time = str("{:.2f}".format(rounded_time)).split('.')
    minutes = int(split_time[0])
    seconds = int(split_time[1][:2])
    return 60 * minutes + seconds


def find_episodes(time, labels, key):
    """ Find the start and end times of all episodes of a behavior or sequence
    referenced to the fluorescence time series.

    Parameters
    ----------
    time : pd.Series or np.array
        Time vector
    labels : list or iterable object of str
        The labeled behaviors or zone occupancies
    key : str
        The episode type to find

    Returns
    -------
        indices and times of an episode occurrence
    """

    if "Zone" in key:
        start_end = labels[[" ".join([key, "In"]), " ".join([key, "Out"])]].dropna().to_numpy()
    else:
        start_end = labels[[" ".join([key, "Start"]), " ".join([key, "End"])]].dropna().to_numpy()

    # Create a vectorized version of get_sec_from_min_sec to apply to whole arrays
    vec_get_sec_from_min_sec = np.vectorize(get_sec_from_min_sec)

    episode_idxs = []
    episode_times = []

    if len(start_end) > 0:

        start_end_behavior = vec_get_sec_from_min_sec(start_end)

        for event in start_end_behavior:
            start_idx, start_time = find_nearest(time, event[0])
            end_idx, end_time = find_nearest(time, event[1])

            episode_idxs.append([start_idx, end_idx])
            episode_times.append([start_time, end_time])

    return episode_idxs, episode_times


def find_zone_and_behavior_episodes(data_df, behavior_labels):
    """ Locates the times at which a given behavior occurs in an experiment

    Parameters
    ----------
    data_df : pd.DataFrame
        The prprocessed fluorescence data
    behavior_labels : iterable object of str
        The column names from the behavior labeling

    Returns
    -------
        The start and end times and indices of a given episode
    """

    ts = data_df['time'].to_numpy()
    behaviors = [' '.join(col.split(' ')[0:-1]) for col in behavior_labels.columns if 'Start' in col.split(' ')[-1]]
    zones = [' '.join(col.split(' ')[0:-1]) for col in behavior_labels.columns if 'In' in col.split(' ')[-1]]

    behav_bouts = []
    for behav in behaviors:

        behav_idxs, behav_times = find_episodes(ts, behavior_labels, behav)

        for idxs, times in zip(behav_idxs, behav_times):
            behav_bouts.append([behav, idxs[0], times[0], idxs[1], times[1]])

    behav_bouts = np.array(behav_bouts)

    zone_bouts = []
    for zone in zones:
        zone_idxs, zone_times = find_episodes(ts, behavior_labels, zone)

        for idxs, times in zip(zone_idxs, zone_times):
            zone_bouts.append([zone, idxs[0], times[0], idxs[1], times[1]])

    zone_bouts = np.array(zone_bouts)

    return behav_bouts, zone_bouts


def add_episode_data(data_df, behavior_bouts, zone_bouts):
    """ Adds labeled behavior data to a dataframe

    Parameters
    ----------
    data_df : pd.DataFrame
    behavior_bouts : tuple or list
        List of behaviors with start and end times and indices
    zone_bouts : tuple or list
        List of zone occupancies with start and end times and indices

    Returns
    -------
        pd.DataFrame with behavior episodes
    """

    behaviors = np.unique(behavior_bouts[:, 0])
    zones = np.unique(zone_bouts[:, 0])

    # Create columns to hold the labels for all behaviors and zone occupations
    data_df['behavior'] = np.array([''] * len(data_df))
    data_df['zone'] = np.array([''] * len(data_df))

    # Process the instances of each labelled behavior
    for behavior in behaviors:
        data_df[behavior] = np.array([''] * len(data_df))

    behavior_df = pd.DataFrame(behavior_bouts, columns=['behavior', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in behavior_df.iterrows():
        index_range = np.arange(int(val['start_idx']), int(val['end_idx']))
        data_df.loc[index_range, val['behavior']] = val['behavior']
        data_df.loc[index_range, 'behavior'] = val['behavior']

    # Process the time spent in each zone
    for zone in zones:
        data_df[zone] = np.array([''] * len(data_df))

    zone_df = pd.DataFrame(zone_bouts, columns=['zone', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in zone_df.iterrows():
        index_range = np.arange(int(val['start_idx']), int(val['end_idx']))
        data_df.loc[index_range, val['zone']] = val['zone']
        data_df.loc[index_range, 'zone'] = val['zone']

    return data_df


def get_mean_episode(episodes):
    """ Gets the mean trace from an array of traces

    Parameters
    ----------
    episodes :

    Returns
    -------

    """

    f_traces = [e[1] for e in episodes]

    trace_array = list_lists_to_array(f_traces)

    mean_trace = np.nanmean(trace_array, axis=0)
    std_trace = np.nanstd(trace_array, axis=0)

    return trace_array, mean_trace, std_trace


def remove_baseline(time, traces, norm_start=-5, norm_end=0):
    """ Removes the baseline of a trace given a specific time window by subtracting the median of that window

    Parameters
    ----------
    time : pd.Series, list, or np.array
        The time trace
    traces : pd.Series, list, or np.array
        The fluorescence trace
    norm_start : int or float
        Start time of the normalization window
    norm_end : int or float
        End time of the normalization window

    Returns
    -------
        traces with baseline subtracted
    """

    start_idx, _ = find_nearest(time, norm_start)
    end_idx, _ = find_nearest(time, norm_end)
    baseline = np.median(traces[:, start_idx:end_idx + 1], axis=-1)
    traces = traces - np.expand_dims(baseline, axis=1)
    return traces


def median_of_time_window(time, trace, t_start, t_end):
    """ Finds the median of a given range of times in a trace

    Parameters
    ----------
    time : pd.Series, list, or np.array
        The time trace
    trace : pd.Series, list, or np.array
        The fluorescence trace
    t_start : int or float
        Start time of the median window
    t_end : int or float
        End time of the median window

    Returns
    -------
        median value of selected trace in specified time window

    """
    # time is assumed to be in seconds

    time_mask = (time >= t_start) & (time <= t_end)
    return np.median(trace[time_mask])


def check_if_dual_channel_recording(data_df):
    """ Checks if a recording is a dual-channel recording or not

    Parameters
    ----------
    data_df : pd.DataFrame

    Returns
    -------
    is_DC : bool
        True if a dual-channel recording, false otherwise
    """
    is_DC = False

    for col in list(data_df.columns):
        if ('anterior' in col) or ('posterior' in col):
            is_DC = True

    return is_DC

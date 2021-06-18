import numpy as np


def find_nearest(array: np.array, value: float):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_lists_to_array(list_of_lists):
    max_length = max([len(sublist) for sublist in list_of_lists])
    new_array = np.empty((len(list_of_lists), max_length))
    new_array[:] = np.NaN

    for row, l in enumerate(list_of_lists):
        new_array[row, :len(l)] = l

    return new_array


def get_sec_from_min_sec(time: int):
    """Converts a float representing time in min.sec to seconds"""
    split_time = str(time).split('.')
    minutes = int(split_time[0])
    seconds = int(split_time[1])
    return 60 * minutes + seconds


def find_episodes(time, labels, key):
    """
    Find the start and end times of all episodes of a behavior or sequence referenced to the fluorescence time series.
    :param time:
    :param labels:
    :param key:
    :return:
    """

    if "Zone" in key:
        start_end = labels[[" ".join([key, "In"]), " ".join([key, "Out"])]].dropna().to_numpy()
    else:
        start_end = labels[[" ".join([key, "Start"]), " ".join([key, "End"])]].dropna().to_numpy()

    # Create a vectorized version of get_sec_from_min_sec to apply to whole arrays
    vec_get_sec_from_min_sec = np.vectorize(get_sec_from_min_sec)

    epidsode_idxs = []
    epidode_times = []

    if len(start_end) > 0:

        start_end_behavior = vec_get_sec_from_min_sec(start_end)

        for event in start_end_behavior:
            start_idx, start_time = find_nearest(time, event[0])
            end_idx, end_time = find_nearest(time, event[1])

            epidsode_idxs.append([start_idx, end_idx])
            epidode_times.append([start_time, end_time])

    return epidsode_idxs, epidode_times


def get_mean_episode(episodes):
    """

    :param episodes:
    :return:
    """

    f_traces = [e[1] for e in episodes]

    trace_array = list_lists_to_array(f_traces)

    mean_trace = np.nanmean(trace_array, axis=0)
    std_trace = np.nanstd(trace_array, axis=0)

    return trace_array, mean_trace, std_trace


# norm.window here is a default; if you don't pass the parameter in the code it will resort to -5
def remove_baseline(time, traces, norm_start=-5, norm_end=0):
    start_idx, _ = find_nearest(time, norm_start)
    end_idx, _ = find_nearest(time, norm_end)
    baseline = np.median(traces[:, start_idx:end_idx], axis=-1)
    traces = traces - np.expand_dims(baseline, axis=1)
    return traces


def median_of_time_window(time, trace, t_start, t_end):
    # time is assumed to be in seconds

    time_mask = (time >= t_start) & (time <= t_end)
    return np.median(trace[time_mask])

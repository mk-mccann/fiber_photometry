import numpy as np
from functions_utils import find_nearest


def subtract_baseline_median(time_trace, f_trace, start_time=None, end_time=None):
    # TODO maybe unused, check if can be removed or not
    if start_time is not None:
        start_idx, _ = find_nearest(time_trace, start_time)
    else:
        start_idx = 0

    if end_time is not None:
        end_idx, _ = find_nearest(time_trace, end_time)
    else:
        end_idx = -1

    baseline = np.median(f_trace[start_idx:end_idx])
    return f_trace - baseline


def median_large_jumps(trace, percentile=0.95):
    """
    Filters large jumps in fluorescence signal and replaces them with the overall median

    :param trace:
    :param percentile:
    :return:
    """

    filtered_trace = trace.copy()

    # TODO: interpolate points where jumps occur instead of using overall median
    med = np.median(trace)    # Take median of whole fluor trace
    mask = np.argwhere((trace < percentile * med))   # Find where trace less than some fraction of median
    filtered_trace[mask] = med

    return filtered_trace


def downsample(ts, signal, ds_factor):
    """

    :param ts:
    :param signal:
    :param ds_factor:
    :return:
    """

    signal_ds = np.mean(np.resize(signal, (int(np.floor(signal.size / ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor / 2)), ts.size, ds_factor)]

    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds


def remove_nans(trace):
    """

    :param trace:
    :return:
    """

    mask = np.isnan(trace)
    trace[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), trace[~mask])
    return trace


def zscore_median(trace):
    """
    Takes the Z-score of a time series trace, using the median of the trace instead of the mean

    :param trace: (np.array) Fluorescence trace
    :return: (np.array) Z-scored fluorescence trace
    """

    return (trace - np.median(trace)) / np.std(trace)


def lernerFit(auto, gcamp, power=1):
    # fitting like in LERNER paper
    # https://github.com/talialerner
    reg = np.polyfit(auto, gcamp, power)
    a = reg[0]
    b = reg[1]
    controlFit = a * auto + b
    return controlFit

import numpy as np
import scipy.signal as sig
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
    
    mask = np.argwhere((trace < med*percentile)) 
    
    std = np.std(trace)
    # # Find where trace is greater than num_std standard deviations above the 
    # num_std=2
    # mask_plus = np.argwhere((trace >= med + num_std*std))   
    # mask_minus = np.argwhere((trace <= med - num_std*std))  
    # mask = np.unique(np.concatenate((mask_plus, mask_minus)))
    
    filtered_trace[mask] = med

    return filtered_trace


def interpolate_large_jumps(trace, percentile=0.95):
    filtered_trace = trace.copy()

    # TODO: interpolate points where jumps occur instead of using overall median
    med = np.median(trace)  # Take median of whole trace
    mask = np.argwhere((trace < percentile * med))  # Find where trace less than some fraction of median
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
    Removes NaNs from an input array by interpolation

    :param trace: (np.array) Input fluorescence trace
    :return: Interpolated fluorescence trace with nan removed
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

    return (trace - np.nanmedian(trace)) / np.nanstd(trace)


def lernerFit(auto, gcamp, power=1):
    # fitting like in LERNER paper
    # https://github.com/talialerner
    reg = np.polyfit(auto, gcamp, power)
    a = reg[0]
    b = reg[1]
    controlFit = a * auto + b
    return controlFit


def butter_highpass(cutoff, order, fs):
    """
    Calculates the coefficients for a Butterworth high pass filter

    :param cutoff: the cutoff frequency in Hz
    :param order: filter order
    :param fs: sampling frequency in Hz
    """

    nyq = 0.5 * fs
    high_cut = cutoff / nyq
    b, a = sig.butter(order, high_cut, btype='highpass')
    return b, a


def butter_lowpass(cutoff, order, fs):
    """
    Calculates the coefficients for a Butterworth low pass filter

    :param cutoff: the cutoff frequency in Hz
    :param order: filter order
    :param fs: sampling frequency in Hz
    """

    nyq = 0.5 * fs
    low_cut = cutoff / nyq
    b, a = sig.butter(order, low_cut, btype='lowpass')
    return b, a


def find_lost_signal(trace):
    """
    Identifies when a signal was lost by searching for locations where the
    signal derivative is zero

    :param trace: [np.array] Signal to be processed
    """

    d_trace = np.r_[0, np.abs(np.diff(trace))]
    d0 = np.where(d_trace == 0)[0]
    return d0

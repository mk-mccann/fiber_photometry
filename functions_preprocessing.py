import numpy as np
import scipy.signal as sig


def median_large_jumps(trace, percentile=0.95):
    """Filters large jumps in a signal and replaces them with the overall signal median
    Depreciated and no longer used.

    Parameters
    ----------
    trace : np.array
        Signal to cleaned up
    percentile : float
        Signal percentile under which to replace data with the median

    Returns
    -------
    filtered_trace : np.array
        Signal with jumps removed
    """

    filtered_trace = trace.copy()
    med = np.median(trace)
    mask = np.argwhere((trace < med*percentile))
    filtered_trace[mask] = med

    return filtered_trace


def downsample(ts, signal, ds_factor):
    """Down-samples a time-series signal by some factor

    Parameters
    ----------
    ts : np.array
        Time stamps
    signal : np.array
        Signal to be down-sampled
    ds_factor : int or float
        Factor by which to down-sample

    Returns
    -------
    Down-sampled time stamps and time-series signal

    """

    signal_ds = np.mean(np.resize(signal, (int(np.floor(signal.size / ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor / 2)), ts.size, ds_factor)]

    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds


def remove_nans(trace):
    """Removes NaNs from an input array by interpolation


    Parameters
    ----------
    trace : np.array
        Input trace containing nans

    Returns
    -------
    trace : np.array
        Interpolated fluorescence trace with nan removed
    """

    mask = np.isnan(trace)
    trace[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), trace[~mask])
    return trace


def zscore_median(trace):
    """Takes the Z-score of a time series trace, using the median of the trace instead of the mean.
    Ignores any NaNs in the trace during calculation.

    Parameters
    ----------
    trace : np.array
        Input signal

    Returns
    -------
    Z-scored signal
    """

    return (trace - np.nanmedian(trace)) / np.nanstd(trace)


def lernerFit(auto, gcamp, power=1):
    """Polynomial fitting of a fiber photometry signal as done in the Lerner paper
    and done here https://github.com/talialerner

    Parameters
    ----------
    auto : np.array or pd.Series object
        Autofluorescence signal
    gcamp : np.array or pd.Series object
        GCaMP signal
    power : int, default=1
        polynomial order to fit

    Returns
    -------
    controlFit : np.array
        Polynomial fitting of the fiber photometry signal
    """

    reg = np.polyfit(auto, gcamp, power)
    a = reg[0]
    b = reg[1]
    controlFit = a * auto + b
    return controlFit


def butter_highpass(cutoff, order, fs):
    """Calculates the coefficients for a Butterworth high pass filter


    Parameters
    ----------
    cutoff : int or float
        -3dB cutoff frequency in Hz
    order : int
        Filter order
    fs : int
        Signal sampling frequency

    Returns
    -------
    b, a :
        Filter coefficients
    """

    nyq = 0.5 * fs
    high_cut = cutoff / nyq
    b, a = sig.butter(order, high_cut, btype='highpass')
    return b, a


def butter_lowpass(cutoff, order, fs):
    """Calculates the coefficients for a Butterworth low pass filter


    Parameters
    ----------
    cutoff : int or float
        -3dB cutoff frequency in Hz
    order : int
        Filter order
    fs : int
        Signal sampling frequency

    Returns
    -------
    b, a :
        Filter coefficients
    """

    nyq = 0.5 * fs
    low_cut = cutoff / nyq
    b, a = sig.butter(order, low_cut, btype='lowpass')
    return b, a


def find_lost_signal(trace):
    """Identifies when a signal was lost by searching for locations where the signal derivative is zero

    Parameters
    ----------
    trace : np.array
        Signal to be processed

    Returns
    -------
    d0 : np.array
        Indices of 'trace' array where the derivative goes to zero
    """

    d_trace = np.r_[0, np.abs(np.diff(trace))]
    d0 = np.where(d_trace == 0)[0]
    return d0

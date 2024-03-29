import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta, datetime
from scipy.stats import sem


# For a list of all MPL colors: https://matplotlib.org/stable/gallery/color/named_colors.html
episode_colors = {'Eating': 'cyan',
                  'Grooming': '#FAC205',
                  'Digging': 'lime',
                  'Tail Suspension': '#800000',
                  'Transfer': '#008000',
                  'WSW': 'indigo',
                  'Squeezed MZ Edge': 'mediumblue',
                  'Social Interaction': '#FF00FF',
                  'Ear Scratch': 'firebrick',
                  'Switch': 'sienna',
                  'Idle': 'silver',
                  'Nesting': 'papayawhip',
                  'Nibbling Floor': 'thistle',
                  'Nibbling Tape': 'aquamarine',
                  'Shock': '#FF0000',
                  'Eating Zone': 'b',
                  'Marble Zone': 'g',
                  'Nesting Zone': 'gray',
                  'Social Interaction Zone': 'teal',
                  'Eating Window': "#00FFFF",
                  'Eating Window DC': "#0343DF",
                  'Air Puff': 'firebrick',
                  'Freezing': 'violet',
                  'Low Activity':'blanchedalmond',
                  'Rearing': 'lightseagreen',
                  'Interspersed Digging': 'steelblue',
                  'Lemon Smell': 'yellow',
                  'Nose Poke': 'yellowgreen',
                  'Squeezed Zone': 'mediumvioletred',
                  'Water Spray': 'deepskyblue',
                  'Peanut Jar': 'sienna',
                  'Jump': 'coral',
                  'Edge Zone': 'brown',
                  "Running": 'brown'
                  }


fluorescence_axis_labels = {'auto_raw': 'Fluorescence (A.U.)',
                            'gcamp_raw': 'Fluorescence (A.U.)',
                            'auto': 'Fluorescence (A.U.)',
                            'gcamp': 'Fluorescence (A.U.)',
                            'dff': 'dF/F',
                            'dff_Lerner': 'dF/F',
                            'zscore': 'Z-dF/F',
                            'zscore_Lerner': 'Z-dF/F',
                            }


def get_mpl_datetime(time):
    """Converts a float representing time in MM.SS format to HH:MM:SS format

    Parameters
    ----------
    time : float or str
        A decimal number indicating time in MM.SS

    Returns
    -------
    Time in matplotlib datetime format
    """

    zero = datetime(2021, 1, 1)

    split_time = str(time).split('.')
    dt = timedelta(minutes=int(split_time[0]), seconds=int(split_time[1]))
    return md.date2num(zero + dt)


# Make sure that all of the times in the csv are in the "Text", rather than a "Number" format
# All the times must be in a two decimal format; always 15.50, never 15.5
def mpl_datetime_from_seconds(time):
    """Takes either an integer or an array and converts it to MPL HH:MM:SS datetime format

    Parameters
    ----------
    time : int or float or iterable object of ints or floats
        Time in seconds

    Returns
    -------
        Time in matplotlib datetime format
    """

    zero = datetime(2021, 1, 1)

    if isinstance(time, int) or isinstance(time, float):
        return md.date2num(zero + timedelta(seconds=time))
    else:
        return [md.date2num(zero + timedelta(seconds=t)) for t in time]


def plot_fluorescence_min_sec(time, trace, ax=None):
    """Plots a time series fluorescence trace with time in HH:MM:SS formatting

    Parameters
    ----------
    time : iterable object of ints or floats
        Time in seconds
    trace : np.array
        Time series fluorescence trace
    ax : Matplotlib axis object, optional
        Axes to current figure, if already generated

    Returns
    -------
    Matplotlib axis to current figure with fluorescence trace plotted

    """

    time_format = mpl_datetime_from_seconds(time)

    if ax is None:
        fig = plt.figure(figsize=(20, 10))
        fig.add_subplot(111)
        ax = plt.gca()

    # Format x axis to take HH:MM:SS format
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()

    ax.plot(time_format, trace)
    ax.set_xlabel('Time')
    return ax


def overlay_manual_episodes(episodes, label, ax):
    """Creates horizontal span overlays for episodes of a scoring type

    Parameters
    ----------
    episodes : iterable object
        List of start and end times of episodes
    label : str
        The name of the scoring type being plotted
    ax : Matplotlib axis object
        Axis of current figure on which to overlay the scoring episode

    Returns
    -------
    labeled_section : Matplotlib axis object
        Matplotlib axis object with overlaid scoring episodes
    """

    for episode in episodes:
        labeled_section = ax.axvspan(mpl_datetime_from_seconds(episode[0]), mpl_datetime_from_seconds(episode[-1]),
                                     facecolor=episode_colors[label],
                                     alpha=0.3)

    return labeled_section


def highlight_episodes(data: pd.DataFrame, keys, ax=None):
    """Highlights the episodes of given by 'keys' on a fluorescence trace

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of an experiment with preprocessed fluorescence and scoring data
    keys : iterable object of strings
        The scoring types to be plotted
    ax : Matplotlib axis object, optional
        Axis of current figure on which to overlay the scoring episode

    Returns
    -------
    Axis of current figure with overlaid scoring episodes

    See Also
    --------
    color_overlay : creates the overlay objects
    """

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    time = data['time'].to_numpy()

    # Create the highlighted episodes
    vspans = []
    for key in keys:
        data_col = data[key].to_numpy()
        episodes_to_plot = data_col == key
        label = color_overlay(mpl_datetime_from_seconds(time), episodes_to_plot, key, ax)
        vspans.append([label, key])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def color_overlay(x, bool_array, label, ax):
    """Creates named and colored overlay sections for a given scoring type

    Parameters
    ----------
    x : list or np.array
        The x-axis data points for th plot
    bool_array : list or np.array of booleans
        Where on the x-axis to overlay color for this scoring type
    label : str
        The scoring type being overlaid
    ax : Matplotlib axis object
        Axis of current figure on which to overlay the scoring episode

    Returns
    -------
    Axis of current figure with overlaid scoring episodes for the scoring type given by 'label'
    """

    labeled_section = ax.fill_between(x, 0, 1, where=bool_array,
                    facecolor=episode_colors[label],
                    alpha=0.5,
                    transform=ax.get_xaxis_transform())

    return labeled_section


def plot_mean_episode(time, traces, plot_singles=False, ax=None, plot_ep_n=False, **kwargs):
    """Plots the mean trace of a scoring type. PLots mean + SEM.

    Parameters
    ----------
    time : list or np.array
        Time vector
    traces : np.array
        Array of all fluorescence traces for episodes of a given scoring type
    plot_singles : bool, default=False
        Plot single episode traces
    ax : Matplotlib axis object, optional
        Axis of current figure on which to overlay the scoring episode

    Returns
    -------
    Matplotlib axis object with mean + SEM plotted
    """

    fill_between_color = kwargs.get('fill_between_color', 'cyan')
    if fill_between_color != 'cyan':
        fill_between_color = episode_colors[fill_between_color]

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(111)
        ax = plt.gca()

    num_episodes = traces.shape[0]
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = sem(traces, axis=0, nan_policy='omit')

    if plot_singles:
        for trace in traces:
            plt.plot(time, trace, c='gray', alpha=0.5)

    plt.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace,
                     alpha=0.5, color=fill_between_color)
    plt.plot(time, mean_trace, c='k', linewidth=2)
    plt.xlim(time[0], time[-1])
    plt.axvline(0, color="orangered")

    # Set the tick labels font
    fontsize = kwargs.get('fontsize', 20.0)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(fontsize)

    if plot_ep_n:
        plt.text(0.05, 0.95, "n = " + str(num_episodes), fontsize=fontsize, transform=plt.gca().transAxes)


    return ax


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmap(values, xlabels, ylabels, ax=None, use_colorbar=True, **kwargs):
    """ Wrapper function to plot a heatmap with a colorbar

    Parameters
    ----------
    values : np.Array
        values to be plotted
    xlabels : iterable of str
        X axis labels
    ylabels : iterable of str
        Y axis labels
    ax : matplotlib AxisObejct, optional
        Axis on which to create the heatmap
    use_colorbar : bool, optional
        Use a colorbar or not
    kwargs
        cmap_min : int or float
            Minimum value for the colormap
        cmap_max : int or float
            Maximum value for the colormap

    Returns
    -------

    """

    if ax is None:
        fig, ax = plt.subplots()

    cmap_min = kwargs.get('cmap_min', 0)
    cmap_max = kwargs.get('cmap_max', np.max(values))

    im = ax.imshow(values, cmap='viridis', vmin=cmap_min, vmax=cmap_max)
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    if use_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)

    return im, ax
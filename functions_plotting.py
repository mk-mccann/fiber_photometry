import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import timedelta, datetime
from scipy.stats import sem
import numpy as np
import pandas as pd

from functions_utils import find_episodes


# For a list of all MPL colors: https://matplotlib.org/stable/gallery/color/named_colors.html


episode_colors = {'Eating': 'cyan',
                  'Grooming': 'goldenrod',
                  'Digging': 'lime',
                  'Transfer': 'forestgreen',
                  'WSW': 'indigo',
                  'Squeezed MZ Edge': 'mediumblue',
                  'Social Interaction': 'deeppink',
                  'Ear Scratch': 'firebrick',
                  'Switch': 'sienna',
                  'Idle': 'silver',
                  'Nibbling Floor': 'thistle',
                  'Nibbling Tape': 'aquamarine',
                  'Shock': 'red',
                  'Eating Zone': 'b',
                  'Marble Zone': 'g',
                  'Nesting Zone': 'gray',
                  }


def get_mpl_datetime(time: float):
    """ Time comes in format min.sec"""
    zero = datetime(2021, 1, 1)

    split_time = str(time).split('.')
    dt = timedelta(minutes=int(split_time[0]), seconds=int(split_time[1]))
    return md.date2num(zero + dt)


# Make sure that all of the times in the csv are in the "Text", rather than a "Number" format
# All the times must be in a two decimal format; always 15.50, never 15.5
def mpl_datetime_from_seconds(time):
    """ Takes either an integer or an array and converts it to mpl datetime format"""
    zero = datetime(2021, 1, 1)

    if isinstance(time, int) or isinstance(time, float):
        return md.date2num(zero + timedelta(seconds=time))
    else:
        return [md.date2num(zero + timedelta(seconds=t)) for t in time]


def plot_fluorescence_min_sec(time, trace, ax=None):
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


def overlay_manual_episodes(epochs, label, ax):
    """

    :param epochs:
    :param label:
    :param ax:
    :return:
    """

    for epoch in epochs:
        labeled_section = ax.axvspan(mpl_datetime_from_seconds(epoch[0]), mpl_datetime_from_seconds(epoch[-1]),
                                     facecolor=episode_colors[label],
                                     alpha=0.3)

    return labeled_section


def highlight_episodes(data: pd.DataFrame, column, keys, ax=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    time = data['time']
    data_col = data[column].to_numpy()

    # Create the highlighted episodes
    vspans = []
    for key in keys:
        episodes_to_plot = data_col == key
        label = color_overlay(mpl_datetime_from_seconds(time), episodes_to_plot, key, ax)
        vspans.append([label, key])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax

def highlight_glm_episodes(time: np.array, glm_predictions: pd.DataFrame, glm_keys, ax=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []
    for col, key in zip(glm_predictions.columns, glm_keys):

        label = color_overlay(mpl_datetime_from_seconds(time), glm_predictions[col].to_numpy(), key, ax)
        vspans.append([label, key])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def color_overlay(x, bool_array, label, ax):

    labeled_section = ax.fill_between(x, 0, 1, where=bool_array,
                    facecolor=episode_colors[label],
                    alpha=0.5,
                    transform=ax.get_xaxis_transform())

    return labeled_section



def plot_mean_episode(time, traces, plot_singles=False, ax=None):

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(111)
        ax = plt.gca()

    num_episodes = traces.shape[0]
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = sem(traces, axis=0, nan_policy='omit')

    if plot_singles:
        for trace in traces:
            plt.plot(time, trace)

    plt.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2)
    plt.plot(time, mean_trace, c='k', linewidth=2)
    # plt.ylim([-0.25, 1.5])
    plt.axvline(0, color="orangered")
    plt.text(0.05, 0.95, "n = " + str(num_episodes), fontsize='large', transform=plt.gca().transAxes)

    plt.xlabel('Time from Behavior Start (s)')

    return ax
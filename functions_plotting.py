import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import timedelta, datetime


def get_mpl_datetime(time: str):
    """ Time comes in format min.sec"""
    zero = datetime(2021, 1, 1)

    split_time = str(time).split('.')
    dt = timedelta(minutes=int(split_time[0]), seconds=int(split_time[1]))
    return md.date2num(zero + dt)


# Make sure that all of the times in the csv are in the "Text", rather than a "Number" format
# All the times must be in a two decimal format; always 15.50, never 15.5
def mpl_datetime_from_seconds(time: int):
    """ Takes either an integer or an array and converts it to mpl datetime format"""
    zero = datetime(2021, 1, 1)

    if isinstance(time, int):
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

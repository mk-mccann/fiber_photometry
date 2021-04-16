import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.signal import medfilt
from scipy.stats import sem

import paths
import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors


def episode_start_window(time, labels, key, period=(-5, 5), dwell_filter=0):
    """
    Wrapper function for functions_utils.find_episodes. Applies a filter to find valid episodes based on the dwell
    time, and returns the window surrounding the start of an episode in both index and time units

    :param time:
    :param labels:
    :param key:
    :param period:
    :param dwell_filter:
    :return: valid_episode_idxs, valid_episode_times
    """

    valid_episode_idxs = []
    valid_episode_times = []

    # Get the starting and ending indexes and times of all episodes of a given type
    _, times = f_util.find_episodes(time, labels, key)

    # These now need to meet our criteria for being a valid episode
    for t in times:
        [start, end] = t
        dwell_time = end - start

        if dwell_time <= dwell_filter:
            start_idx, start_time = f_util.find_nearest(time, start + period[0])
            end_idx, end_time = f_util.find_nearest(time, start + period[1])

            valid_episode_idxs.append([start_idx, end_idx])
            valid_episode_times.append([start_time, end_time])

    return valid_episode_idxs, valid_episode_times


def plot_mean_episode(time, traces, plot_singles=False):
    fig = plt.figure(figsize=(10, 10))

    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = sem(traces, axis=0, nan_policy='omit')

    if plot_singles:
        for trace in traces:
            plt.plot(t, trace)

    plt.fill_between(time, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2)
    plt.plot(time, mean_trace, c='k', linewidth=2)
    # plt.ylim([-0.25, 1.5])
    plt.axvline(0, color="orangered")
    plt.text(0.05, 0.95, "n = " + str(num_episodes), fontsize='large', transform=plt.gca().transAxes)

    plt.xlabel('Time from Behavior Start (s)')

    return fig


if __name__ == "__main__":

    summary_file_path = paths.summary_file    # Set this to wherever it is
    save_directory = paths.figure_directory   # Set this to wherever you want
    f_io.check_dir_exists(save_directory)

    # Read the summary file as a pandas dataframe
    all_exps = f_io.read_summary_file(summary_file_path)
    
    # remove certain days
    exps_to_run = all_exps
    #exps_to_run = all_exps.loc[all_exps["Day"] == 3]

    # Which behavior do you want to look at
    key = 'ALL'    # TODO If set to "ALL", generates means for all behaviors
    period = (-5, 10)
    dfilt = 30

    if key == "ALL":
        all_episodes = {k: [] for k in episode_colors.keys()}
    else:
        all_episodes = {key: []}

    # Go row by row through the summary data
    for idx, row in exps_to_run.iterrows():

        try:
            # load the raw data from 1 rec at a time
            labels = f_io.load_behavior_labels(str(row['Ani_ID']))
            data = f_io.load_preprocessed_data(str(row['Ani_ID']))
            # labels = f_io.load_behavior_labels(str(row['Ani_ID']), base_directory=row['Behavior Labelling'])
            # data = f_io.load_preprocessed_data(str(row['Ani_ID']), base_directory=row['Preprocessed Data'])

            dffzscore = medfilt(data['zscore'], kernel_size=51)

            for k in all_episodes.keys():
                try:
                    window_idxs, window_times = episode_start_window(data['time'], labels, k, period=period, dwell_filter=dfilt)
                    exp_episodes = [dffzscore[start:end] for [start, end] in window_idxs]
                    all_episodes[k].append(exp_episodes)
                except KeyError:
                    continue
        
        except FileNotFoundError as error:
            print(str(error))

    # Loop through all the conditions we pulled out before and plot them
    for k in all_episodes.keys():
        f_traces_of_key = all_episodes[k]
        f_traces_of_key = f_util.flatten_list(f_traces_of_key)
        #all_traces_of_key = list(filter(None, all_traces_of_key))
        #f_traces = [e[1] for e in all_episodes]

        # TODO look at this
        trace_array = f_util.list_lists_to_array(f_traces_of_key)

        num_episodes = trace_array.shape[0]
        print("Number of {} trials = {}".format(k, num_episodes))

        t = np.linspace(period[0], period[1], trace_array.shape[-1])
        trace_array = f_util.remove_baseline(t, trace_array, norm_window=-5)

        fig = plot_mean_episode(t, trace_array)
        plt.ylabel('$\Delta$F/F Z-score minus')
        plt.title('Mean trace for {}'.format(k))
        plt_name = "mean_{}_dff_zscore.png".format(key.lower())
        # plt.savefig(join(save_directory, plt_name))
        plt.show()
    


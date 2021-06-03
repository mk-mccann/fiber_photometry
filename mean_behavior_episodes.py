import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.signal import medfilt

import paths
import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors, plot_mean_episode


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

        # Dwell filter: animal must spend at least this much time doing a behavior
        if dwell_time >= dwell_filter:
            start_idx, start_time = f_util.find_nearest(time, start + period[0])
            end_idx, end_time = f_util.find_nearest(time, start + period[1])

            valid_episode_idxs.append([start_idx, end_idx])
            valid_episode_times.append([start_time, end_time])

    return valid_episode_idxs, valid_episode_times


if __name__ == "__main__":

    summary_file_path = paths.summary_file    # Set this to wherever it is
    save_directory = paths.figure_directory   # Set this to wherever you want
    f_io.check_dir_exists(save_directory)

    # Read the summary file as a pandas dataframe
    all_exps = f_io.read_summary_file(summary_file_path)
    
    # remove certain days
    exps_to_run = all_exps
    # exps_to_run = all_exps.loc[all_exps["Day"] == 3]

    # Which behavior do you want to look at
    key = 'ALL'    # If set to "ALL", generates means for all behaviors
    period = (-5, 10)

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

            time = data['ts']
            f_trace = data['zscore']
            f_trace = medfilt(f_trace, kernel_size=51)

            for k in all_episodes.keys():
                if 'Eating' in k:
                    # Dwell time filter can only be applied for eating!
                    dwell_filt = 30  # Dwell time filter: minimum time animal must stay in zone
                else:
                    dwell_filt = 0

                try:
                    window_idxs, window_times = episode_start_window(time, labels, k, period=period, dwell_filter=dwell_filt)
                    exp_episodes = [f_trace[start:end] for [start, end] in window_idxs]
                    all_episodes[k].append(exp_episodes)
                except KeyError:
                    continue
        
        except FileNotFoundError as error:
            print(str(error))

    # Loop through all the conditions we pulled out before and plot them
    for k in all_episodes.keys():
        f_traces_of_key = all_episodes[k]

        if len(f_traces_of_key) > 0:
            f_traces_of_key = f_util.flatten_list(f_traces_of_key)
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} trials = {}".format(k, num_episodes))

            t = np.linspace(period[0], period[1], trace_array.shape[-1])
            trace_array = f_util.remove_baseline(t, trace_array, norm_window=-5)

            fig = plot_mean_episode(t, trace_array, plot_singles=True)
            plt.ylabel('$\Delta$F/F Z-score minus')
            plt.title('Mean trace for {}'.format(k))
            plt_name = "mean_{}_dff_zscore.png".format(k.lower())
            plt.savefig(join(save_directory, plt_name))
            plt.show()

        else:
            print("No episodes of {} found!".format(k))

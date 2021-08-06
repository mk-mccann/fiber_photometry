import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
import warnings
from scipy.signal import medfilt

import paths
import functions_utils as f_util
import functions_io as f_io
from functions_plotting import episode_colors, plot_mean_episode


def get_individual_episode_indices(data_df, key):
    """


    :param data_df:
    :param key:
    :return: valid_episode_idxs
    """

    valid_episode_idxs = []

    # Get the starting and ending indexes and times of all episodes of a given type
    if 'Zone' in key:
        # Here is a special case for the eating zone. We only want to look at times in the eating zone when the mouse
        # is actually eating
        if 'Eating' in key:
            episode_idxs = (data_df[(data_df['zone'] == key) & (data_df['behavior'] == 'Eating')].index.to_numpy())
        else:
            episode_idxs = data_df[data_df['zone'] == key].index.to_numpy()
    else:
        episode_idxs = data_df[data_df['behavior'] == key].index.to_numpy()

    # episodes_to_plot is a list of all indices fulfilling the behavior or zone occupancy condition.
    # We want specific episodes, so split the list. If there are no episodes of a given behavior, return
    # an empty list
    if episode_idxs.size > 0:
        breaks = np.where(np.diff(episode_idxs) != 1)[0] + 1  # add 1 to compensate for the diff
        valid_episode_idxs = np.array_split(episode_idxs, breaks)
    else:
        pass

    return valid_episode_idxs


def check_episode_duration(data_df, episodes, min_dwell_time=0):
    valid_episodes = []

    for ep in episodes:
        start_idx = ep[0]
        end_idx = ep[-1]
        start_time = data_df['time'].iloc[start_idx]
        end_time = data_df['time'].iloc[end_idx]
        dwell_time = end_time - start_time

        # Dwell filter: animal must spend at least this much time doing a behavior
        if dwell_time >= min_dwell_time:
            valid_episodes.append(ep)

    return valid_episodes


def get_episode_start_window(data_df, episodes, window_period=(-5, 5)):

    start_windows = []

    # Some variables to number the episodes of a behavior within a given experiment
    ep_number = 1
    last_animal = 0
    last_day = 0

    for ep in episodes:
        ep_start_idx = ep[0]

        # For getting the start windows, we need to look at the experiment that a given episode occurred in.
        animal = data_df['animal'].iloc[ep_start_idx]
        day = data_df['day'].iloc[ep_start_idx]
        exp = data_df.loc[(data_df['animal'] == animal) & (data_df['day'] == day)]

        # Check to see if this animal or day changed from the last episode in the list. If so, reset counter.
        # Otherwise advance the counter
        if (int(animal) != last_animal) or (int(day) != last_day):
            ep_number = 1
        else:
            ep_number = ep_number + 1

        ep_start_time = exp['time'][exp.index == ep_start_idx].item()

        _, window_start_time = f_util.find_nearest(exp['time'], ep_start_time + window_period[0])
        _, window_end_time = f_util.find_nearest(exp['time'], ep_start_time + window_period[1])

        try:
            start_idx = exp.loc[exp['time'] == window_start_time].index.item()
        except IndexError:
            start_idx = exp.loc[exp['time'] == exp['time'].min()].index.item()

        try:
            end_idx = exp.loc[exp['time'] == window_end_time].index.item()
        except IndexError:
            end_idx = exp.loc[exp['time'] == exp['time'].max()].index.item()

        ep_df = data_df.iloc[start_idx:end_idx].copy()
        ep_df['episode_number'] = ep_number
        start_windows.append(ep_df)

        last_animal = int(animal)
        last_day = int(day)

    return start_windows


def take_first_n_episodes(episodes, n_to_keep=-1):
    episodes_to_keep = []

    if n_to_keep == -1:
        episodes_to_keep = episodes
    else:
        for ep in episodes:
            episode_number = int(ep['episode_number'].iloc[0])

            if episode_number <= n_to_keep:
                episodes_to_keep.append(ep)

    return episodes_to_keep


def load_all_experiments():
    """
    Loads all experiments that have been scored and preprocessed into a giant dataframe

    :return: (pd.DataFrame) DataFrame of all experiments
    """

    # Read the summary file as a pandas dataframe
    all_exps = f_io.read_summary_file(paths.summary_file)

    df_list = []

    # Go row by row through the summary data to create a GIANT dataframe with all experiments that are preprocessed and
    # are scored
    for idx, row in all_exps.iterrows():
        # Get identifying info about the experiment
        animal, day = str(row['Ani_ID']).split(".")

        try:
            # load the processed data from one experiment at a time
            exp = f_io.load_preprocessed_data(animal, day)

            # Some error catching - if the behavior data is not in the df, raise an error and go to the next experiment
            if 'behavior' not in exp.columns:
                warnings.warn('Behavior labeling not present in DataFrame. Trying to load now.')
                try:
                    behavior_labels = f_io.load_behavior_labels(animal, day)
                    behavior_bouts, zone_bouts = f_util.find_zone_and_behavior_episodes(exp, behavior_labels)
                    exp = f_util.add_episode_data(exp, behavior_bouts, zone_bouts)
                except FileNotFoundError as err:
                    print("Manual scoring needs to be done for this experiment: Animal {} Day {}. \n{}\n".format(
                        animal, day, err))
                    continue

        except FileNotFoundError as error:
            print(str(error))
            continue

        # If the selected dataframe is good, add it to the list
        df_list.append(exp)

    # Now create a giant dataframe from all of the experiments
    return pd.concat(df_list).reset_index(drop=True)


def plot_individual_behaviors(data_dict, f_trace='zscore', plot_singles=False):
    """

    :param data_dict:
    :param f_trace:
    :param plot_singles:
    :return:
    """

    # Loop through all the conditions we pulled out before and plot them
    for k in data_dict.keys():
        f_traces_of_key = [df[f_trace].to_numpy() for df in data_dict[k]]

        if len(f_traces_of_key) > 0:
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} episodes = {}".format(k, num_episodes))

            t = np.linspace(period[0], period[1], trace_array.shape[-1])

            # Remove the baseline using the 5 seconds before behavior onset
            trace_array = f_util.remove_baseline(t, trace_array)

            # Plot the mean episode
            fig = plot_mean_episode(t, trace_array, plot_singles=plot_singles)
            plt.ylabel('Z-dF/F')
            plt.title('Mean trace for {}'.format(k))
            plt_name = "mean_{}_dff_zscore.png".format(k.lower())
            plt.savefig(join(paths.figure_directory, plt_name))

        else:
            print("No episodes of {} found!".format(k))


def plot_multiple_behaviors(data_dict, keys_to_plot, f_trace='zscore', plot_singles=False):
    collected_traces = []

    for k in keys_to_plot:
        f_traces_of_key = [df[f_trace].to_numpy() for df in data_dict[k]]

        if len(f_traces_of_key) > 0:
            trace_array = f_util.list_lists_to_array(f_traces_of_key)

            num_episodes = trace_array.shape[0]
            print("Number of {} episodes = {}".format(k, num_episodes))

            t = np.linspace(period[0], period[1], trace_array.shape[-1])

            # Remove the baseline using the 5 seconds before behavior onset
            trace_array = f_util.remove_baseline(t, trace_array, norm_start=-13, norm_end=-10)

            collected_traces.append(trace_array)

        else:
            print("No episodes of {} found!".format(k))

    collected_traces = np.vstack(collected_traces)
    # Plot the mean episode
    fig = plot_mean_episode(t, collected_traces, plot_singles=plot_singles)
    plt.ylabel('Z-dF/F')
    plt.title('Mean trace for {}'.format(', '.join(keys_to_plot)))
    plt_name = "mean_{}_dff_zscore.png".format('_'.join(keys_to_plot))
    plt.savefig(join(paths.figure_directory, plt_name))
    return fig


def extract_episodes(data_df, analysis_period, episodes_to_analyze, extract_first_n=-1):

    # Create a dictionary to hold the output traces
    if episodes_to_analyze == "ALL":
        output_dict = {e: [] for e in episode_colors.keys()}
    else:
        output_dict = {e: [] for e in episodes_to_analyze}

    # Further median filtering for smoothing
    # f_trace = medfilt(f_trace, kernel_size=35)

    # Loop through each of the behaviors you are interested in analyzing
    # in order to extract them.
    for behav in output_dict.keys():
        # Finds individual episodes of a given behavior
        episodes_by_idx = get_individual_episode_indices(data_df, behav)

        # This is a check if there are episodes for a given behavior. If there are, it adds them to the output dict
        if len(episodes_by_idx) > 0:
            # Check the duration of each episode, and throw out any that are too short. Default is to keep everything.
            episodes_by_duration = check_episode_duration(data_df, episodes_by_idx)
            # Extracts a window surrounding the start of an episode, as given by the variable window_period
            ep_start_windows = get_episode_start_window(data_df, episodes_by_duration, window_period=analysis_period)
            # Take first n episodes of a behavior from each experiment. Default is to keep all
            first_n_eps = take_first_n_episodes(ep_start_windows, n_to_keep=extract_first_n)
            output_dict[behav] = ep_start_windows
        else:
            print('No episodes of {} found!'.format(behav))

    return output_dict


if __name__ == "__main__":

    # Read the summary file as a giant pandas dataframe
    all_exps = load_all_experiments()

    # remove certain days/animals
    # exps_to_run = all_exps.loc[all_exps["day"] == 3]    # select day 3 exps
    # exps_to_run = all_exps.loc[all_exps["animal"] != 1]    # remove animal 1
    exps_to_run = all_exps

    # Which behavior(s) do you want to look at?
    # If set to "ALL", generates means for all behaviors.
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', ...]
    behaviors_to_analyze = 'ALL'
    period = (-13, 10)

    # The first n episodes of each behavior to keep. Setting this value to -1 keeps all episodes
    # If you only wanted to keep the first two, use
    # first_n_eps = 2
    first_n_eps = -1

    # Run the main function
    all_episodes = extract_episodes(exps_to_run, period, behaviors_to_analyze, extract_first_n=first_n_eps)

    # Check if the figure-saving directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Plot means for the individual behaviors (as selected in behaviors_to_analyze)
    plot_individual_behaviors(all_episodes)

    # Plot means across all or some of the behaviors
    # If you set key = 'ALL' initially, and want to just look at a subset of behaviors,
    # then you need change multi_behav_plot with a list of the behaviors you want
    # to see. The example below plots all behaviors, but no zone occupancies
    multi_behav_plot = [key for key in list(all_episodes.keys()) if 'Zone' not in key]
    plot_multiple_behaviors(all_episodes, multi_behav_plot)
    plt.show()

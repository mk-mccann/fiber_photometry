import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.signal import medfilt
from scipy.stats import sem

import paths
import functions_utils as f_util
import functions_io as f_io


def find_episodes(time, f_trace, labels, key, period=(-5, 5), dwell_filter=0):
    exp_episodes = []
     
    if "Zone" in key:
        start_end = labels[[" ".join([key, "In"]), " ".join([key, "Out"])]].dropna().to_numpy()
    else:
        start_end = labels[[" ".join([key, "Start"]), " ".join([key, "End"])]].dropna().to_numpy()

    if len(start_end) > 0:
        vfunc = np.vectorize(f_util.get_sec_from_min_sec)
        start_end = vfunc(start_end)

        for episode in start_end:

            dwell_time = episode[1] - episode[0]

            if dwell_time <= dwell_filter:
                start_idx, start_time = f_util.find_nearest(time, episode[0] + period[0])
                end_idx, end_time = f_util.find_nearest(time, episode[0] + period[1])
                 
                exp_episodes.append([time[start_idx:end_idx], f_trace[start_idx:end_idx]])

    return exp_episodes


def get_mean_episode(episodes):
    f_traces = [e[1] for e in episodes]

    trace_array = f_util.list_lists_to_array(f_traces)
    
    mean_trace = np.nanmean(trace_array, axis=0)
    std_trace = np.nanstd(trace_array, axis=0)
    
    return trace_array, mean_trace, std_trace


# norm.window here is a default; if you don't pass the parameter in the code lower down it will resort to -5
def remove_baseline(time, traces, norm_window=-5):
    idx, _ = f_util.find_nearest(time, 0)
    wind_idx, _ = f_util.find_nearest(time, norm_window)
    baseline = np.median(traces[:, wind_idx:idx], axis=1)
    traces = traces - np.expand_dims(baseline, axis=1)
    return traces


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
    plt.text(-4.5, 0.3, "n = " + str(num_episodes), fontsize='large')

    plt.xlabel('Time from Behavior Start (s)')
    plt.title('Mean trace for {}'.format(key_to_plot))

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
    key = 'Eating Zone'    # TODO If set to "ALL", generates means for all behaviors
    period = (-5, 10)
    dfilt = 30
    all_episodes = []

    # Go row by row through the summary data
    for idx, row in exps_to_run.iterrows():

        try:
            # load the raw data from 1 rec at a time
            labels = f_io.load_behavior_labels(str(row['Ani_ID']))
            data = f_io.load_preprocessed_data(str(row['Ani_ID']))
            # labels = f_io.load_behavior_labels(str(row['Ani_ID']), base_directory=row['Behavior Labelling'])
            # data = f_io.load_preprocessed_data(str(row['Ani_ID']), base_directory=row['Preprocessed Data'])

            dffzscore = medfilt(data['zscore'], kernel_size=51)

            if key == "ALL":
                keys_to_plot = [" ".join(col.split(" ")[0:-1]) for col in labels.columns if
                                "Start" in col.split(" ")[-1]]
            else:
                keys_to_plot = [key]

            exp_episodes = {}
            for k in keys_to_plot:
                exp_episodes[key] = find_episodes(data['time'], dffzscore, labels, k, period=period,
                                             dwell_filter=dfilt)

            all_episodes.append(exp_episodes)
        
        except FileNotFoundError as error:
            print(str(error))

    # TODO get this to work for dict
    all_episodes = f_util.flatten_list(all_episodes)
    all_episodes = list(filter(None, all_episodes))
    f_traces = [e[1] for e in all_episodes]
    trace_array = f_util.list_lists_to_array(f_traces)
    
    num_episodes = trace_array.shape[0]
    print("Number of {} trials = {}".format(key_to_plot, num_episodes))
    
    t = np.linspace(period[0], period[1], trace_array.shape[-1])
    trace_array = remove_baseline(t, trace_array, norm_window=-5)

    fig = plot_mean_episode(t, trace_array)
    plt.ylabel('$\Delta$F/F Z-score minus')
    plt_name = "mean_{}_dff_zscore.png".format(key_to_plot.lower())
    plt.savefig(join(save_directory, plt_name))
    plt.show()
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from scipy.signal import savgol_filter
from tqdm import tqdm

import paths
import functions_preprocessing as fpp
import functions_io as f_io
from functions_plotting import plot_fluorescence_min_sec



"""load and process data for fiber photometry experiments"""


def preprocess_fluorescence(data_df):

    # replace NaN's with closest (interpolated) non-NaN
    gcamp = fpp.remove_nans(data_df['gcamp'].to_numpy())
    auto = fpp.remove_nans(data_df['auto'].to_numpy())

    # replace large jumps with the overall median
    auto = fpp.median_large_jumps(auto)
    gcamp = fpp.median_large_jumps(gcamp)

    # smoothing the data by applying filter
    auto = savgol_filter(auto, 21, 2)
    gcamp = savgol_filter(gcamp, 21, 2)

    # fitting like in LERNER paper
    controlFit = fpp.lernerFit(auto, gcamp)
    # dff = (gcamp - controlFit) / controlFit
    
    # Compute DFF
    dff = (gcamp - auto) / auto
    dff = dff * 100
    
    # zscore whole data set with overall median
    dffzscore = fpp.zscore_median(dff)
    
    # Remove homecage period baseline
    # dff_rem_base = fpp.subtract_baseline_median(fp_times, gcamp, start_time=0, end_time=240)
    # dff_rem_base = dff_rem_base * 100

    data_df['gcamp'] = gcamp
    data_df['auto'] = auto
    data_df['dff'] = dff
    data_df['zscore'] = dffzscore

    return data_df


def find_zone_and_behavior_episodes(data_df, behavior_labels):
    ts = data_df['time']
    behaviors = [" ".join(col.split(" ")[0:-1]) for col in behavior_labels.columns if "Start" in col.split(" ")[-1]]
    zones = [" ".join(col.split(" ")[0:-1]) for col in behavior_labels.columns if "In" in col.split(" ")[-1]]

    behav_bouts = []
    for behav in behaviors:

        behav_idxs, behav_times = f_io.find_episodes(ts, behavior_labels, behav)

        for idxs, times in zip(behav_idxs, behav_times):
            behav_bouts.append([behav, idxs[0], times[0], idxs[1], times[1]])

    behav_bouts = np.array(behav_bouts)

    zone_bouts = []
    for zone in zones:
        zone_idxs, zone_times = f_io.find_episodes(ts, behavior_labels, zone)

        for idxs, times in zip(zone_idxs, zone_times):
            zone_bouts.append([zone, idxs[0], times[0], idxs[1], times[1]])

    zone_bouts = np.array(zone_bouts)

    return behav_bouts, zone_bouts


def add_episode_data(data_df, behav_bouts, zone_bouts):
    behaviors = np.unique(behav_bouts.T[0])
    zones = np.unique(np.unique(zone_bouts.T[0]))

    for zone in zones:
        bool_array = np.array([False] * len(data_df))
        data_df[zone] = bool_array

    zones = pd.DataFrame(zone_bouts, columns=['zone', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in zones.iterrows():
        data_df.loc[val['start_idx']:val['end_idx'], val['zone']] = True

    for behavior in behaviors:
        bool_array = np.array([False] * len(data_df))
        data_df[behavior] = bool_array

    behaviors = pd.DataFrame(behav_bouts, columns=['behavior', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in behaviors.iterrows():
        data_df.loc[val['start_idx']:val['end_idx'], val['behavior']] = True

    return data_df


if __name__ == "__main__":
    summary_file_path = paths.summary_file  # Set this to wherever it is
    output_directory = paths.processed_data_directory  # Set this to wherever you want
    f_io.check_dir_exists(output_directory)

    # Read the summary file as a pandas dataframe
    summary = f_io.read_summary_file(summary_file_path)

    # Go row by row through the summary data
    # tqdm only creates a progress bar for the loop
    for idx, row in tqdm(summary.iterrows(), total=len(summary)):

        # load the raw fluorescence data from a given experiment
        fp_file = join(paths.csv_directory, row['FP file'] + '.csv')
        data = f_io.read_fiber_photometry_csv(fp_file, row)
        data = preprocess_fluorescence(data)

        animal, day = str(row['Ani_ID']).split(".")
        data['animal'] = animal
        data['day'] = day

        # try to load the manual video scoring file, if it exists
        try:
            behavior_labels = f_io.load_behavior_labels(animal, day)
        except FileNotFoundError:
            continue

        # save the data as a .h5 file
        filename = 'Animal{}_Day{}_preprocessed.h5'.format(animal, day)
        data.to_hdf(join(output_directory, filename), key='preprocessed', mode='w')

        ax = plot_fluorescence_min_sec(data['time'], data['zscore'])
        ax.set_title('Animal {} Day {} Z-Score DFF'.format(animal, day))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-Score DFF')
        plt.savefig(join(output_directory, 'Animal{}_Day{}_gcamp_zscore.png'.format(animal, day)), format="png")
        # plt.show()


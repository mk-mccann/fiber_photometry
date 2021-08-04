import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from scipy.signal import savgol_filter
from tqdm import tqdm

import paths
import functions_preprocessing as fpp
import functions_io as f_io
from functions_utils import find_zone_and_behavior_episodes, add_episode_data
from functions_plotting import plot_fluorescence_min_sec


"""load and process data for fiber photometry experiments"""


def preprocess_fluorescence(data_df, channel_key=None):
    """
    Performs preprocessing on raw fiber photometry time series data.
    The user must specify which key defines the gcamp and autofluorescence channels
    to be processed from the dataframe. This lets the function handle multiple
    auto/gcamp channels
    .
    Example usage for a single fiber recording:
    data = preprocess_fluorescence(data)

    Example usage for dual fiber recording:
    data = preprocess_fluorescence(data, 'anterior')
    data = preprocess_fluorescence(data, 'posterior')
    """

    # Define the gcamp and autofluorescence channels
    if channel_key is None:
        auto_channel = data_df['auto']
        gcamp_channel = data_df['gcamp']
    else:
        auto_channel = data_df['auto_' + channel_key]
        gcamp_channel = data_df['gcamp_' + channel_key]

    # replace NaN's with closest (interpolated) non-NaN
    gcamp = fpp.remove_nans(gcamp_channel.to_numpy())
    auto = fpp.remove_nans(auto_channel.to_numpy())

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
    zdff = fpp.zscore_median(dff)

    # Remove homecage period baseline
    # dff_rem_base = fpp.subtract_baseline_median(fp_times, gcamp, start_time=0, end_time=240)
    # dff_rem_base = dff_rem_base * 100

    # Save the data
    if channel_key is None:
        data_df['gcamp'] = gcamp
        data_df['auto'] = auto
        data_df['dff'] = dff
        data_df['zscore'] = zdff
    else:
        data_df['gcamp_' + channel_key] = gcamp
        data_df['auto_' + channel_key] = auto
        data_df['dff_' + channel_key] = dff
        data_df['zscore_' + channel_key] = zdff

    return data_df


if __name__ == "__main__":
    summary_file_path = paths.summary_file  # Set this to wherever it is
    f_io.check_dir_exists(paths.processed_data_directory)

    # Read the summary file as a pandas dataframe
    summary = f_io.read_summary_file(summary_file_path)

    # Go row by row through the summary data
    # tqdm only creates a progress bar for the loop
    for idx, row in tqdm(summary.iterrows(), total=len(summary)):

        # Get identifying info about the experiment
        animal, day = str(row['Ani_ID']).split(".")

        # load the raw fluorescence data from a given experiment
        fp_file = join(paths.csv_directory, row['FP file'] + '.csv')
        data = f_io.read_1_channel_fiber_photometry_csv(fp_file, row)

        # Add the identifying information to the dataframe
        data['animal'] = animal
        data['day'] = day

        # Preprocess the fluorescence with the given channels
        data = preprocess_fluorescence(data)

        # Try to load the manual video scoring file, if it exists.
        # If so, process it. Raise a warning if not.
        try:
            behavior_labels = f_io.load_behavior_labels(animal, day)
            behavior_bouts, zone_bouts = find_zone_and_behavior_episodes(data, behavior_labels)
            data = add_episode_data(data, behavior_bouts, zone_bouts)

        except FileNotFoundError:
            tqdm.write("Manual scoring needs to be done for Animal {} Day {}.".format(animal, day))
            continue

        # save the data as an .h5 file
        filename = 'animal{}_day{}_preprocessed.h5'.format(animal, day)
        data.to_hdf(join(paths.processed_data_directory, filename), key='preprocessed', mode='w')

        # Make a plot of the zdff and save it.
        ax = plot_fluorescence_min_sec(data['time'], data['zscore'])
        ax.set_title('Animal {} Day {} Z-dF/F'.format(animal, day))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-dF/F')
        plt.savefig(join(paths.figure_directory, 'animal{}_day{}_gcamp_zscore.png'.format(animal, day)), format="png")
        # plt.show()


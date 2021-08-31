import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.signal import savgol_filter, filtfilt
from tqdm import tqdm
import pathlib

import paths
import functions_preprocessing as fpp
import functions_io as f_io
from functions_utils import find_zone_and_behavior_episodes, add_episode_data
from functions_plotting import plot_fluorescence_min_sec


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

    # Define the gcamp and autofluorescence channels and save a copy of the raw
    # data in a new column
    if channel_key is None:
        auto_channel = data_df['auto']
        gcamp_channel = data_df['gcamp']

        data_df['auto_raw'] = auto_channel.copy()
        data_df['auto_raw'] = gcamp_channel.copy()

    else:
        auto_channel = data_df['auto_' + channel_key]
        gcamp_channel = data_df['gcamp_' + channel_key]

        data_df['auto_' + channel_key + '_raw'] = auto_channel.copy()
        data_df['gcamp_' + channel_key + '_raw'] = gcamp_channel.copy()

    # replace NaN's with closest (interpolated) non-NaN
    gcamp = fpp.remove_nans(gcamp_channel.to_numpy())
    auto = fpp.remove_nans(auto_channel.to_numpy())

    # identify where signal is lost -  we remove this from later traces
    gcamp_d0 = fpp.find_lost_signal(gcamp)
    auto_d0 = fpp.find_lost_signal(auto)
    shared_zero = np.unique(np.concatenate((gcamp_d0, auto_d0)))

    # remove slow decay with a high pass filter
    cutoff = 0.1    # Hz
    order = 3
    fs = 40         # Hz
    b, a = fpp.butter_highpass(cutoff, order, fs)
    gcamp = filtfilt(b, a, gcamp)
    auto = filtfilt(b, a, auto)

    # smooth data and remove noise with a low pass filter
    cutoff = 19    # Hz
    d, c = fpp.butter_lowpass(cutoff, order, fs)
    gcamp = filtfilt(d, c, gcamp)
    auto = filtfilt(d, c, auto)

    # smoothing the data by applying filter
    gcamp = savgol_filter(gcamp, 21, 2)
    auto = savgol_filter(auto, 21, 2)

    # fitting like in LERNER paper
    controlFit = fpp.lernerFit(auto, gcamp)
    # dff = (gcamp - controlFit) / controlFit

    # Compute DFF
    dff = (gcamp - auto) / auto
    dff = dff * 100

    # z-score whole data set with overall median
    zdff = fpp.zscore_median(dff)

    # Remove homecage period baseline
    # dff_rem_base = fpp.subtract_baseline_median(fp_times, gcamp, start_time=0, end_time=240)
    # dff_rem_base = dff_rem_base * 100

    # Remove sections where the signal is lost
    gcamp[shared_zero] = np.NaN
    auto[shared_zero] = np.NaN
    dff[shared_zero] = np.NaN
    zdff[shared_zero] = np.NaN

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


def plot_single_channel(data):
    animal = data['animal'].unique()
    day = data['day'].unique()

    ax = plot_fluorescence_min_sec(data['time'], data['zscore'])

    ax.set_title('Animal {} Day {} Z-dF/F'.format(animal, day))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-dF/F')

    plt.savefig(join(paths.figure_directory, 'animal{}_day{}_gcamp_zscore.png'.format(animal, day)), format="png")
    return ax


def plot_dual_channel(data):
    test_number = data['test'].unique()

    # Make a plot of the zdffs and save it.
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 15))

    # Plot the anterior recording
    ax1 = plot_fluorescence_min_sec(data['time'], data['zscore_anterior'], ax=ax1)
    ax1.set_ylabel('Z-dF/F')
    ax1.set_title('Anterior')

    # Plot the posterior recording
    ax2 = plot_fluorescence_min_sec(data['time'], data['zscore_posterior'], ax=ax2)
    ax2.set_ylabel('Z-dF/F')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Posterior')

    fig.suptitle('Test {} dual channel Z-dF/F'.format(test_number))

    # Save the figure
    plt.savefig(join(paths.figure_directory, 'test{}_dual_rec_gcamp_zscore.png'.format(test_number)), format="png")
    return fig


if __name__ == "__main__":
    # TODO: This script is incomplete. Needs a standardized file naming convention before it can be done

    # Check to see if the processed data and figure directories exist
    f_io.check_dir_exists(paths.processed_data_directory)
    f_io.check_dir_exists(paths.figure_directory)

    # Get the list of files to preprocess, or link to the summary file sheet
    files = list(pathlib.Path(paths.dual_recording_csv_directory).glob('*.csv'))

    # Go row by row through the summary data
    # tqdm only creates a progress bar for the loop
    for file in tqdm(files):

        test_number = int(file.stem[-1])

        data = f_io.read_2_channel_fiber_photometry_csv(file.resolve())

        # Preprocess the fluorescence with the given channels
        data = preprocess_fluorescence(data, 'anterior')
        data = preprocess_fluorescence(data, 'posterior')

        # # Try to load the manual video scoring file, if it exists.
        # # If so, process it. Raise a warning if not.
        # try:
        #     behavior_labels = f_io.load_behavior_labels(animal, day)
        #     behavior_bouts, zone_bouts = find_zone_and_behavior_episodes(data, behavior_labels)
        #     data = add_episode_data(data, behavior_bouts, zone_bouts)
        #
        # except FileNotFoundError:
        #     print("Manual scoring needs to be done for Animal {} Day {}.".format(animal, day))
        #     continue

        # Add the identifying information to the dataframe
        data['test'] = int(test_number)

        # save the data as an .h5 file
        filename = 'test{}_dual_rec_preprocessed.h5'.format(test_number)
        data.to_hdf(join(paths.processed_data_directory, filename), key='preprocessed', mode='w')

        # Make a plot of the zdffs and save it.
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 15), sharex=False)

        # Plot the anterior recording
        ax1 = plot_fluorescence_min_sec(data['time'], data['zscore_anterior'], ax=ax1)
        ax1.set_ylabel('Z-dF/F')
        ax1.set_title('Anterior')

        # Plot the posterior recording
        ax2 = plot_fluorescence_min_sec(data['time'], data['zscore_posterior'], ax=ax2)
        ax2.set_ylabel('Z-dF/F')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Posterior')

        fig.suptitle('Test {} dual channel Z-dF/F'.format(test_number))

        plt.savefig(join(paths.figure_directory, 'test{}_dual_rec_gcamp_zscore.png'.format(test_number)), format="png")
        plt.show()

import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
from scipy.signal import savgol_filter, filtfilt
from scipy.ndimage import percentile_filter
from tqdm import tqdm

# Import custom written functions
import paths
import functions_preprocessing as fpp
import functions_io as f_io
from functions_plotting import plot_fluorescence_min_sec


"""
load and process data for fiber photometry experiments. Saves preprocessed
fluorescence data as an .h5 file.
"""


def preprocess_fluorescence(data_df, channel_key=None):
    """
    Performs preprocessing on raw fiber photometry time series data.
    The user must specify which key defines the gcamp and autofluorescence channels
    to be processed from the dataframe. This lets the function handle multiple
    auto/gcamp channels
    .
    Example usage for a single fiber recording:
    data = preprocess_fluorescence(data) - no need to specify channel_key

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
        data_df['gcamp_raw'] = gcamp_channel.copy()

    else:
        auto_channel = data_df['auto_' + channel_key]
        gcamp_channel = data_df['gcamp_' + channel_key]

        data_df['auto_' + channel_key + '_raw'] = auto_channel.copy()
        data_df['gcamp_' + channel_key + '_raw'] = gcamp_channel.copy()

    # replace NaN's with closest (interpolated) non-NaN
    gcamp = fpp.remove_nans(gcamp_channel.to_numpy())
    auto = fpp.remove_nans(auto_channel.to_numpy())

    # Remove jumps in the data with an aggressive percentile filter
    gcamp = percentile_filter(gcamp, percentile=99, size=25)
    auto = percentile_filter(auto, percentile=99, size=25)

    # identify where signal is lost -  we remove this from later traces
    # gcamp_d0 = fpp.find_lost_signal(gcamp)
    # auto_d0 = fpp.find_lost_signal(auto)
    # shared_zero = np.unique(np.concatenate((gcamp_d0, auto_d0))) # identifies the indices if signal is lost in at least one channel

    # remove slow decay with a high pass filter
    # cutoff = 0.1    # Hz
    # order = 3
    # fs = 40         # Hz
    # b, a = fpp.butter_highpass(cutoff, order, fs)
    # gcamp = filtfilt(b, a, gcamp)
    # auto = filtfilt(b, a, auto)
    
    # # smooth data and remove noise with a low pass filter
    # cutoff = 19    # Hz
    # d, c = fpp.butter_lowpass(cutoff, order, fs)
    # gcamp = filtfilt(d, c, gcamp)
    # auto = filtfilt(d, c, auto)
    
    # # remove large jumps by replacing with the median
    # gcamp = fpp.median_large_jumps(gcamp)
    # auto = fpp.median_large_jumps(auto)
    
    # smoothing the data by applying filter
    gcamp = savgol_filter(gcamp, 21, 2)
    auto = savgol_filter(auto, 21, 2)

    # Compute DFF
    dff = (gcamp - auto) / auto
    dff = dff * 100

    # Remove homecage period baseline
    # If you want to use this, comment out the dff calculation above
    # dff = fpp.subtract_baseline_median(fp_times, gcamp, start_time=0, end_time=240)
    # dff = dff * 100

    # z-score whole data set with overall median
    zdff = fpp.zscore_median(dff)

    # fitting like in LERNER paper (calcuated but not used)
    controlFit = fpp.lernerFit(auto, gcamp)
    dff = (gcamp - controlFit) / controlFit
    zdff = fpp.zscore_median(dff)

    # # Remove sections where the signal is lost
    # gcamp[shared_zero] = np.NaN
    # auto[shared_zero] = np.NaN
    # dff[shared_zero] = np.NaN
    # zdff[shared_zero] = np.NaN

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
    # Check that output data directories are present
    f_io.check_dir_exists(paths.processed_data_directory)
    f_io.check_dir_exists(paths.figure_directory)

    # Get a list of all files in the raw data directory
    files = list(Path(paths.raw_data_directory).glob('*.csv'))

    # tqdm only creates a progress bar for the loop through all the raw data files
    for file in tqdm(files):

        # Get identifying info about the experiment
        animal = file.stem.split('_')[-1]
        day = file.stem.split('_')[0][-1]

        # load the raw fluorescence data from a given experiment
        data = f_io.load_1_channel_fiber_photometry_csv(file.resolve())

        # Add the identifying information to the dataframe
        data['animal'] = animal
        data['day'] = day

        # Preprocess the fluorescence with the given channels
        data = preprocess_fluorescence(data)

        # save the data as an .h5 file
        filename = 'animal{}_day{}_preprocessed.h5'.format(animal, day)
        data.to_hdf(join(paths.processed_data_directory, filename), key='preprocessed', mode='w')

        # Make a plot of the zdff and save it.
        ax = plot_fluorescence_min_sec(data['time'], data['gcamp'])
        ax.set_title('Animal {} Day {} Z-dF/F'.format(animal, day))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-dF/F')
        plt.savefig(join(paths.figure_directory, 'animal{}_day{}_gcamp_zscore.png'.format(animal, day)), format="png")
        # plt.show()


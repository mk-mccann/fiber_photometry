import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from scipy.signal import savgol_filter
from tqdm import tqdm

import paths
import functions_preprocessing as fpp
from functions_plotting import plot_fluorescence_min_sec
from functions_io import read_summary_file, read_fiber_photometry_csv, check_dir_exists


"""load and process data for fiber photometry experiments"""


def preprocess_fluorescence(time, gcamp, auto):

    # replace NaN's with closest (interpolated) non-NaN
    gcamp = fpp.remove_nans(gcamp)
    auto = fpp.remove_nans(auto)

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

    return auto, gcamp, dff, dffzscore


if __name__ == "__main__":
    summary_file_path = paths.summary_file  # Set this to wherever it is
    output_directory = paths.processed_data_directory  # Set this to wherever you want
    check_dir_exists(output_directory)

    # Read the summary file as a pandas dataframe
    summary = read_summary_file(summary_file_path)

    # Go row by row through the summary data
    # tqdm only creates a progress bar for the loop
    for idx, row in tqdm(summary.iterrows(), total=len(summary)):
        data = {}

        # load the raw data from 1 rec at a time
        # fp_file = join(row['Raw Data Folder'], row['FP file'] + '.csv')
        fp_file = join(paths.csv_directory, row['FP file'] + '.csv')
        time, auto, gcamp = read_fiber_photometry_csv(fp_file, row)

        auto, gcamp, dff, dffzscore, = preprocess_fluorescence(gcamp, auto)

        data['ani_id'] = row['Ani_ID']
        data['time'] = time
        data['auto'] = auto
        data['gcamp'] = gcamp
        data['dff'] = dff
        data['zscore'] = dffzscore

        # save dictionaries using numpy
        # File format is save_directory/Ani_ID.npy
        # TODO change saving to h5 files
        np.save(join(output_directory, str(row['Ani_ID']) + '.npy'), data)

        fig, axes = plt.subplots(3, 1)
        ax = plot_fluorescence_min_sec(data['time'], data['zscore'], ax=axes[0])
        ax1 = plot_fluorescence_min_sec(data['time'], data['dff'], ax=axes[1])
        ax2 = plot_fluorescence_min_sec(data['time'], data['gcamp'] - data['auto'], ax=axes[2])
        ax.set_title(str(row['Ani_ID']) + " Z-Score DFF")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-Score DFF')
        plt.savefig(join(output_directory, str(row['Ani_ID']) + '_gcamp_ts.png'), format="png")
        # plt.show()


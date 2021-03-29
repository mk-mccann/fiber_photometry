import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import paths
import function_preprocessing as fpp
from functions_plotting import plot_fluorescence_min_sec
from functions_io import read_summary_file, read_fiber_photometry_csv



#load and process data for fiber photometry experiments
#smooth and fit data with auto signal


def load_data_FP(exp_metadata):

    fp_file = os.path.join(exp_metadata['Raw Data Folder'], exp_metadata['FP file'] + '.csv')

    # load the corresponding fp file (ignore the first raw with text)
    time, auto, gcamp = read_fiber_photometry_csv(fp_file, exp_metadata)

    # replace NaN's with closest non-NaN
    gcamp = fpp.remove_nans(gcamp)
    auto = fpp.remove_nans(auto)

    # replace large jumps with the median
    auto = fpp.median_large_jumps(auto)
    gcamp = fpp.median_large_jumps(gcamp)

    # compute the dff, adapted from Alex's matlab code
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

    return fp_times, auto, gcamp, dff, dffzscore


# find CS bool - boolean of CS duration
def find_CS_bool(ts, start_times_CSplus, end_times_CSplus):
    CS_bool = np.zeros(ts.size, dtype=bool)
    preCS_bool = np.zeros(ts.size, dtype=bool)
    postCS_bool = np.zeros(ts.size, dtype=bool)
    for j in np.arange(start_times_CSplus.size):
        start_CS_ind = np.argmin(np.abs(ts - start_times_CSplus[j]))
        end_CS_ind = np.argmin(np.abs(ts - end_times_CSplus[j]))
        CS_bool[start_CS_ind:end_CS_ind] = True
        start_preCS_ind = np.argmin(np.abs(ts - (start_times_CSplus[j] - 30)))
        end_preCS_ind = np.argmin(np.abs(ts - (end_times_CSplus[j] - 30)))
        preCS_bool[start_preCS_ind:end_preCS_ind] = True
        start_postCS_ind = np.argmin(np.abs(ts - (start_times_CSplus[j] + 30)))
        end_postCS_ind = np.argmin(np.abs(ts - (end_times_CSplus[j] + 30)))
        postCS_bool[start_postCS_ind:end_postCS_ind] = True

    return CS_bool, preCS_bool, postCS_bool


def tsplotSlice(corrData, shockStartTimepoints, windowPlusMinus):
    counter = 0
    # tempDf1 = pd.DataFrame()
    tempDf1 = []
    for i in shockStartTimepoints:
        temp1 = corrData[(i - windowPlusMinus): (i + windowPlusMinus)]
        tempDf1.append(temp1)

        counter = counter + 1

    return np.array(tempDf1)


if __name__ == "__main__":

    summary_file_path = paths.summary_file    # Set this to wherever it is
    save_directory = paths.processed_data_directory    # Set this to wherever you want

    # Read the summary file as a pandas dataframe
    all_data = read_summary_file(summary_file_path)

    # Go row by row through the summary data
    for idx, row in all_data.iterrows():
        data = {}

        # load the raw data from 1 rec at a time
        fp_times, auto, gcamp, dff, dffzscore, = load_data_FP(row)

        data['ani_id'] = row['Ani_ID']
        data['ts'] = fp_times
        data['auto'] = auto
        data['gcamp'] = gcamp
        data['dff'] = dff
        data['zscore'] = dffzscore

        # save dictionaries using numpy
        # File format is save_directory/Ani_ID.npy
        # I would suggest using a different file format like hdf5, but this is
        # fine for now.
        np.save(os.path.join(save_directory, str(row['Ani_ID']) + '.npy'), data)

        ax = plot_fluorescence_min_sec(data['ts'], data['zscore'])

        ax.set_title(str(row['Ani_ID']) + " Z-Score DFF")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-Score DFF')
        plt.savefig(os.path.join(save_directory, str(row['Ani_ID']) + '_gcamp_ts.png'), format="png")
        # plt.show()

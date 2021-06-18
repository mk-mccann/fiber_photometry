import numpy as np
import matplotlib.pyplot as plt

import paths
import functions_io as f_io
from functions_utils import median_of_time_window


if __name__ == "__main__":

    mouse_ID = 2
    day = 1
    id = "{}.{}".format(mouse_ID, day)

    dff_dir = paths.processed_data_directory
    save_directory = paths.figure_directory
    f_io.check_dir_exists(save_directory)

    data = f_io.load_preprocessed_data(id)

    # time window is assumed to be in seconds
    # the last two values are the start and end times you want to take the median from
    median_home_cage = median_of_time_window(data['time'], data['zscore'], 0, 300)
    median_middle = median_of_time_window(data['time'], data['zscore'], 1000, 1300)

    print('Median of the home cage period: {}'.format(median_home_cage))
    print('Median of the middle period: {}'.format(median_middle))

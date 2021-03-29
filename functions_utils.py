import numpy as np
import pandas as pd


def find_nearest(array: np.array, value: float):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_lists_to_array(list_of_lists):
    max_length = max([len(l) for l in list_of_lists])
    new_array = np.empty((len(list_of_lists), max_length))
    new_array[:] = np.NaN

    for row, l in enumerate(list_of_lists):
        new_array[row, :len(l)] = l

    return new_array


def get_sec_from_min_sec(time: int):
    """Converts a float representing time in min.sec to seconds"""
    split_time = str(time).split('.')
    minutes = int(split_time[0])
    seconds = int(split_time[1])
    return 60 * minutes + seconds


def decimate_csv(file_path, step=300):
    # Load the csv
    df = pd.read_csv(file_path, skiprows=1, header=0)

    # The initial temporal offset is given roughly by this index.
    # Not sure what exactly this is, but it matches the other files
    initial_unit = 1048

    # Drop these first indices and reset the index
    df = df.drop(range(initial_unit)).reset_index(drop=True)

    time_vector = df[['Time(s)']].to_numpy()[::step]
    df2 = df.drop('Time(s)', axis=1)

    # Get the means by step
    mean_df = df2.groupby(df2.index // step).mean()

    # Add time back
    mean_df = mean_df.insert(0, 'Time(s)', time_vector).copy()

    return mean_df

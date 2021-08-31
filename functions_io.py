import pandas as pd
import os

import paths


def read_summary_file(file_path):
    """
    Reads the Excel file containing paths for all experimental data and the correct columns to read FP data

    :param file_path: (str) Path to the summary .xlsx file
    :return: Pandas DataFrame containing all the information in the summary file.
    """

    summary_file = pd.ExcelFile(file_path, engine='openpyxl')
    sheets = list(summary_file.sheet_names)
    sheet_dfs = []

    for sheet in sheets:
        df = pd.read_excel(summary_file, header=0, sheet_name=sheet, dtype=object, engine='openpyxl')
        day = int(sheet[-1])
        df['Day'] = day
        sheet_dfs.append(df)

    all_data = pd.concat(sheet_dfs)

    return all_data


def load_behavior_labels(animal, day, base_directory=paths.behavior_scoring_directory):
    """
    Loads the Excel file with the behavior labels.

    Takes the animal ID and directory containing the files as inputs.

    :param animal: (int) Animal ID number
    :param day: (int) Experimental day number
    :param base_directory: (str) Path where processed data is located
    :return: Pandas DataFrame of the manually labelled behaviors
    """

    label_filename = r"ID{}_Day{}.xlsx".format(animal, day)
    x = pd.ExcelFile(os.path.join(base_directory, label_filename), engine='openpyxl')
    behavior_labels = pd.read_excel(x, header=0, dtype=object, engine='openpyxl')
    return behavior_labels


def load_preprocessed_data(animal, day, key="preprocessed", base_directory=paths.processed_data_directory):
    """
    Loads the .h5 file of the processed data.

    Assumes the naming convention for the files follows the
    animal#_day# format. Returns loaded data as a pandas DataFrame.

    :param animal: (int) Animal ID number
    :param day: (int) Experimental day number
    :param key: (str) Key to open h5 file. Currently set ot "preprocessed"
    :param base_directory: (str) Path where processed data is located
    :return: Pandas DataFrame of the processed data
    """

    filename = 'animal{}_day{}_preprocessed.h5'.format(animal, day)
    preproc_data = pd.read_hdf(os.path.join(base_directory, filename), key=key)
    return preproc_data


def read_1_channel_fiber_photometry_csv(file_path, file_metadata=None, column_names=None):
    """
    Reads the CSV containing fiber photometry data

    :param file_path: (str) Path to the raw data .csv
    :param file_metadata: (pd.DataFrame) A row from a pandas DataFrame with the relevant columns to extract
    :param column_names: (list) A list of strings with column names. Default is a NoneType.
    :return: Pandas DataFrame with of the time-series fluorescence data.
    """

    # We assume that we are only creating a dataframe to hold a single channel 
    # of fluorescence data (time, gcamp, auto), so here the columns names are 
    # set to a default.
    if column_names is None:
        column_names = ['time', 'auto', 'gcamp']

    if file_metadata is None:
        columns_to_use = [0, 1, 3]
    else:
        columns_to_use = [file_metadata['ts'], file_metadata['gcamp column'], file_metadata['auto column']]

    fluor_df = pd.read_csv(file_path, skiprows=2, names=column_names, 
    usecols=columns_to_use,
    dtype=float
    )

    return fluor_df


def read_2_channel_fiber_photometry_csv(file_path, column_names=None):
    """
    Reads the CSV containing fiber photometry data, but for two simultaneous recordings in a single animal

    :param file_path: (str) Path to the raw data .csv
    :param column_names: (list) A list of strings with column names. Default is a NoneType.
    :return: Pandas DataFrame with of the time-series fluorescence data.
    """

    if column_names is None:
        column_names = ['time', 'auto_anterior', 'gcamp_anterior', 'auto_posterior', 'gcamp_posterior']

    fluor_df = pd.read_csv(file_path, skiprows=2, names=column_names, usecols=[0, 1, 2, 4, 5], dtype=float)

    return fluor_df


def load_glm_h5(filename, key='nokey', base_directory=paths.modeling_data_directory):
    """
    Loads the .h5 file containing the GLM predictions

    :param filename: (str) Filename of the GLM .h5 file
    :param key: (str) Key to access the data in the .h5 file. Default is 'nokey'.
    :param base_directory: (str) Path to the modeling data directory
    :return: (pd.DataFrame) Loaded GLM predition dataframe
    """

    data = pd.read_hdf(os.path.join(base_directory, filename), key=key)
    return data


def check_dir_exists(path):
    """
    Checks if a given directory exists. If not, it creates it.

    :param path: (str) The path to check
    """

    if not os.path.exists(path):
        print('Creating directory: {}'.format(path))
        os.mkdir(path)

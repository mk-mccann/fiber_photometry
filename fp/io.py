import pandas as pd
import os
from warnings import warn
from pathlib import Path

import paths
import fp.utils as f_util


def read_summary_file(file_path):
    """Reads the Excel file containing paths for all experimental data and the correct columns to read FP data.
    Depreciated, as the summary file is no longer used.

    Parameters
    ----------
    file_path : str of PathObject
        Path to the summary .xlsx file

    Returns
    -------
    all_data : pd.DataFrame
        DataFrame containing all the information in the summary file.
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
    """Loads the Excel file with the behavior labels.

    Parameters
    ----------
    animal : int or float or str
        Animal ID number
    day : int or float or str
        Experimental day number
    base_directory : str or PathObject, optional
        Path where processed data is located

    Returns
    -------
    behavior_labels : pd.DataFrame
        DataFrame of the manually labelled behaviors

    """

    label_filename = r"ID{}_Day{}.xlsx".format(animal, day)
    x = pd.ExcelFile(os.path.join(base_directory, label_filename), engine='openpyxl')
    behavior_labels = pd.read_excel(x, header=0, dtype=object, engine='openpyxl')
    return behavior_labels


def check_preprocessed_df_for_scoring(exp_df, animal, day):
    """Checks if the preprocessed fluorescence DataFrame has the behavior data loaded into it

    Parameters
    ----------
    exp_df : pd.DataFrame
        The preprocessed fluorescence data
    animal : int or float or str
        Animal ID number
    day : int or float or str
        Experimental day number

    Returns
    -------
    exp_df : pd.DataFrame
        Preprocessed fluorescence data with behavior labels

    """

    if 'behavior' not in exp_df.columns:
        warn('Behavior labeling not present in DataFrame. Trying to load now...')
   
        behavior_labels = load_behavior_labels(animal, day)
        behavior_bouts, zone_bouts = f_util.find_zone_and_behavior_episodes(exp_df, behavior_labels)
        exp_df = f_util.add_episode_data(exp_df, behavior_bouts, zone_bouts)
        
        return exp_df

    else:
        return exp_df


def load_preprocessed_data(animal, day, key="preprocessed", base_directory=paths.preprocessed_data_directory):
    """Loads the .h5 file with the preprocessed fluorescence data.

    Parameters
    ----------
    animal : int or float or str
        Animal ID number
    day : int or float or str
        Experimental day number
    key : str, default='preprocessed'
        Key to open the .h5 file
    base_directory : str or PathObject, optional
        Path where processed data is located

    Returns
    -------
    preprocessed_data : pd.DataFrame
        DataFrame of the preprocessed data

    """

    filename = 'animal{}_day{}_preprocessed.h5'.format(animal, day)
    preprocessed_data = pd.read_hdf(os.path.join(base_directory, filename), key=key)
    return preprocessed_data


def load_1_channel_fiber_photometry_csv(file_path, columns_to_use=None, column_names=None):
    """Reads the CSV containing fiber photometry data for single channel recordings

    Parameters
    ----------
    file_path : str or PathObject
        Path to the raw data .csv
    columns_to_use : iterable object, optional
        Columns to extract from the csv
    column_names : iterable object, optional
        Names for the extracted columns of the csv

    Returns
    -------
    fluor_df : pd.DataFrame
        DataFrame of the time-series fluorescence data.
    """

    # We assume that we are only creating a dataframe to hold a single channel 
    # of fluorescence data (time, gcamp, auto), so here the columns names are 
    # set to a default.
    if column_names is None:
        column_names = ['time', 'auto_raw', 'gcamp_raw']

    if columns_to_use is None:
        columns_to_use = [0, 1, 3]

    fluor_df = pd.read_csv(file_path,
                           skiprows=2, names=column_names, usecols=columns_to_use, dtype=float)

    return fluor_df


def load_2_channel_fiber_photometry_csv(file_path, column_names=None):
    """Reads the CSV containing fiber photometry data, but for two simultaneous recordings in a single animal


    Parameters
    ----------
    file_path : str or PathObject
        Path to the raw data .csv
    column_names : iterable object, optional
        Names for the extracted columns of the c

    Returns
    -------
    fluor_df : pd.DataFrame
        DataFrame of the time-series fluorescence data.
    """

    if column_names is None:
        column_names = ['time', 'auto_anterior_raw', 'gcamp_anterior_raw',
                        'auto_posterior_raw', 'gcamp_posterior_raw']

    fluor_df = pd.read_csv(file_path, skiprows=2, names=column_names, usecols=[0, 1, 2, 4, 5], dtype=float)

    return fluor_df


def load_glm_h5(filename, key='nokey', base_directory=paths.modeling_data_directory):
    """Load the .h5 file with GLM classification for a given experiment

    Parameters
    ----------
    filename : str or PathObject
        Filename of the GLM .h5 file
    key : str, default='nokey'
        Key to access the data in the .h5 file.
    base_directory : str or PathObject, optional
        Path to the modeling data directory

    Returns
    -------
    data : pd.DataFrame
        DataFrame with the GLM classification
    """

    data = pd.read_hdf(os.path.join(base_directory, filename), key=key)
    return data


def load_all_experiments(base_directory=paths.preprocessed_data_directory):
    """Loads all experiments that have been scored and preprocessed into a giant dataframe

    Returns
    -------
    DataFrame of all experiments with preprocessed fluorescence data and scoring
    """

    # First check if we have an aggregate file
    if os.path.isfile(os.path.join(base_directory, 'aggregate_all_experiments.h5')):
        # If we have it, load it and move on
        all_exps_df = pd.read_hdf(os.path.join(base_directory, 'aggregate_all_experiments.h5'), key='all_exps')

    else:
        # if we don't have an aggregate file, load from all processed files
        all_exps = list(Path(base_directory).glob('*preprocessed.h5'))

        df_list = []

        # Create a GIANT dataframe with all experiments that are preprocessed and are scored
        for file in all_exps:
            # Get identifying info about the experiment
            animal = file.stem.split('_')[0][6:]
            day = file.stem.split('_')[1][-1]

            try:
                # load the processed data from one experiment at a time
                exp = load_preprocessed_data(animal, day)

                # Some error catching - if the behavior data is not in the df, raise an error and go to the next experiment
                try:
                    exp = check_preprocessed_df_for_scoring(exp, animal, day)
                except FileNotFoundError as err:
                    print("Manual scoring needs to be done for this experiment: Animal {} Day {}. \n{}\n".format(
                        animal, day, err))
                    continue

            except FileNotFoundError as error:
                print(str(error))
                continue

            # If the selected dataframe is good, add it to the list
            df_list.append(exp)

        # Now create a giant dataframe from all of the experiments
        all_exps_df = pd.concat(df_list).reset_index(drop=True)

    return all_exps_df


def save_pandas_dict_to_h5(input_dict, filename, base_directory=paths.preprocessed_data_directory):
    """ Saves a dictionary to am .h5 file where each dictionary key is a key in the .h5 file

    Parameters
    ----------
    input_dict : dict
    filename : str or PathObject
    base_directory : str or PathObject, optional
        Path to the preprocessed data directory

    """

    store = pd.HDFStore(os.path.join(base_directory, filename))

    for key in input_dict:
        data = input_dict[key]
        if isinstance(data, list):
            continue
        else:
            store_name = key.lower().replace(' ', '_')
            store.put(store_name, data)

    store.close()


def load_aggregated_episodes(store, key):
    """Read the .h5 file of aggregated episodes and return the requested store

    Parameters
    ----------
    store : HDFObject
        The loaded .h5 file
    key : str
        The key to the dataframe to be loaded

    Returns
    -------
        pd.DataFrame
    """

    key = key.lower.replace(' ', '_')
    return store.get(key)


def check_dir_exists(path):
    """
    Checks if a given directory exists. If not, it creates it.

    :param path: (str) The path to check
    """

    if not os.path.exists(path):
        print('Creating directory: {}'.format(path))
        os.mkdir(path)

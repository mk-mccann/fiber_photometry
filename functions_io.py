import pandas as pd
import numpy as np
import os

import paths


def read_summary_file(file_path):
    """Reads the Excel file containing paths for all experimental data and the correct columns to read FP data"""

    summary_file = pd.ExcelFile(file_path, engine='openpyxl')
    sheets = list(summary_file.sheet_names)
    sheet_dfs = []

    for sheet in sheets:
        df = pd.read_excel(summary_file, header=0, sheet_name=sheet, dtype=object, engine='openpyxl')
        day = int(sheet[-1])
        df['Day'] = [day for i in range(len(df))]
        sheet_dfs.append(df)

    all_data = pd.concat(sheet_dfs)

    return all_data


def load_behavior_labels(animal_id, base_directory=paths.behavior_scoring_directory):
    """Loads the Excel file with the behavior labels. Takes the animal ID and directory
       containing the files as inputs."""

    label_filename = r"ID{}_Day{}.xlsx".format(*str(animal_id).split('.'))
    x = pd.ExcelFile(os.path.join(base_directory, label_filename), engine='openpyxl')
    behavior_labels = pd.read_excel(x, header=0, dtype=object, engine='openpyxl')
    return behavior_labels


def load_preprocessed_data(animal_id, base_directory=paths.processed_data_directory):
    preprocessed_data_path = base_directory
    preprocessed_filename = r"{}.npy".format(animal_id)
    preproc_data = np.load(os.path.join(preprocessed_data_path, preprocessed_filename), allow_pickle=True)
    return preproc_data.item()


def read_fiber_photometry_csv(file_path, file_metadata, column_names=None):
    """Reads the CSV containing fiber photometry data"""

    # We assume that we are only creating a dataframe to hold a single channel 
    # of fluorescence data (time, gcamp, auto), so here the columns names are 
    # set to a default. If you have more channels in one recording, this function 
    # will need to be updated
    if column_names is None:
        column_names = ['time', 'gcamp', 'auto']

    fluor_df = pd.read_csv(file_path, skiprows=2, names=column_names, 
    usecols=[file_metadata['ts'], file_metadata['gcamp column'], file_metadata['auto column']],
    dtype=float
    )

    return fluor_df


def load_glm_h5(filename, key='nokey', base_directory=paths.modeling_data_directory):
    # filename = r"{}.h5".format(animal_id)
    data = pd.read_hdf(os.path.join(base_directory, filename), key=key)
    return data


def check_dir_exists(path):
    if not os.path.exists(path):
        print('Creating directory: {}'.format(path))
        os.mkdir(path)

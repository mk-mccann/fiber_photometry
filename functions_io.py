import pandas as pd
import numpy as np
from os.path import join


def read_summary_file(file_path):

    summary_file = pd.ExcelFile(file_path)
    sheets = list(summary_file.sheet_names)
    sheet_dfs = []

    for sheet in sheets:
        df = pd.read_excel(summary_file, header=0, sheet_name=sheet, dtype=object)
        day = int(sheet[-1])
        df['Day'] = [day for i in range(len(df))]
        sheet_dfs.append(df)

    all_data = pd.concat(sheet_dfs)
    return all_data


def load_from_excel_summary(exp_metadata):
    ani_id = str(exp_metadata['Ani_ID'])

    behavior_label_path = exp_metadata['Behavior Labelling']
    behavior_filename = r"ID{}_Day{}.xlsx".format(*ani_id.split('.'))
    x = pd.ExcelFile(join(behavior_label_path, behavior_filename))
    behavior_labels = pd.read_excel(x, header=0, dtype=object)

    preprocessed_data_path = exp_metadata['Preprocessed Data']
    dff_filename = r"{}.npy".format(ani_id)               # change path
    data = np.load(join(preprocessed_data_path, dff_filename), allow_pickle=True)
    data = data.item()

    return data, behavior_labels


def read_fiber_photometry_csv(file_path, metadata):
    df = pd.read_csv(file_path, skiprows=2)  # , skiprows = (0), skipfooter =(10))
    time = df.values[:, metadata['ts']]
    autofluorescence = df.values[:, metadata['auto column']].astype(float)
    gcamp = df.values[:, metadata['gcamp column']].astype(float)

    return time, autofluorescence, gcamp
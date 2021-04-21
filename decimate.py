from os.path import join
from pandas import read_csv

from paths import csv_directory


def decimate_csv(file_path, step=300):
    # Load the csv
    df = read_csv(file_path, skiprows=1, header=0)

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


if __name__ == "__main__":
    mouse_ID = 1
    day = 3

    filename = r"Day{}_{}_nondecimated.csv".format(day, mouse_ID)
    save_filename = r"Day{}_{}.csv".format(day, mouse_ID)

    # Decimate and save as csv
    decimated = decimate_csv(join(csv_directory, filename))
    decimated.to_csv(join(csv_directory, save_filename), index=False)

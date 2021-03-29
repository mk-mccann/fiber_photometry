from os.path import join

from paths import csv_directory
from functions_utils import decimate_csv

mouse_ID = 1
day = 3

filename = r"Day{}_{}_nondecimated.csv".format(day, mouse_ID)
save_filename = r"Day{}_{}.csv".format(day, mouse_ID)

# Decimate and save as csv
decimated = decimate_csv(join(csv_directory, filename))
decimated.to_csv(join(csv_directory, save_filename), index=False)

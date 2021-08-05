import matplotlib.pyplot as plt
import pandas as pd
import sys

import functions_io as f_io
import paths

from functions_plotting import plot_fluorescence_min_sec

from PyQt5.QtWidgets import QFileDialog, QApplication
from datetime import timedelta, datetime


def mouse_move(event):
    x, y = event.xdata, event.ydata
    print(x, y)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    filePath, _ = QFileDialog.getOpenFileName()
    print("Loaded: " + filePath)

    # Check that the figure directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Load the preprocesed data
    data = pd.read_hdf(filePath, key='preprocessed')

    fig = plt.figure(figsize=(20, 10))
    plot_fluorescence_min_sec(data['time'], data['dff'])
    plt.ylabel('dF/F')
    plt.show()
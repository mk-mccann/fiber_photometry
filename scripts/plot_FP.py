import matplotlib.pyplot as plt
import pandas as pd
import sys
from os.path import join

import fp.io as f_io
import paths

from fp.visualization import plot_fluorescence_min_sec

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

    # fig = plt.figure(figsize=(20, 10))
    # plot_fluorescence_min_sec(data['time'], data['dff'])
    # plt.ylabel('dF/F')

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    axes[0][0].plot(data.time, data.gcamp_anterior)
    axes[0][0].set_title('Anterior GCaMP')
    axes[0][1].plot(data.time, data.gcamp_posterior)
    axes[0][1].set_title('Posterior GCaMP')
    axes[1][0].plot(data.time, data.auto_anterior)
    axes[1][0].set_title('Anterior Auto')
    axes[1][1].plot(data.time, data.auto_posterior)
    axes[1][1].set_title('Posterior Auto')

    plt.savefig(join(paths.figure_directory, 'test{}_dual_rec_raw.png'.format(filePath.split('test')[1][0])),
                format="png")
    # plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %matplotlib qt

import paths
import functions_plotting as fp
import functions_io as f_io
from functions_utils import find_episodes


def highlight_episodes(time: np.array, labels: pd.DataFrame, tf_array=None, plot_key="all_behaviors", ax=None):
    """

    :param time:
    :param labels:
    :param plot_key:
    :param ax:
    :return:
    """

    # TODO scratching
    # for other in others:
    #     start_instance = labels_df[[other]].dropna().to_numpy()
    #     instance_idx, instance_time = find_nearest(time_HMS, get_mpl_datetime(start_instance))

    # Searches the dataframe for each behavior type, zone type, or other action
    if plot_key == "all_behaviors":
        episodes_to_plot = [" ".join(col.split(" ")[0:-1]) for col in labels.columns if "Start" in col.split(" ")[-1]]
    elif plot_key == "all_zones":
        episodes_to_plot = [" ".join(col.split(" ")[0:-1]) for col in labels.columns if "In" in col.split(" ")[-1]]
    else:
        episodes_to_plot = plot_key

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []
    for episode_type in episodes_to_plot:

        _, epochs = find_episodes(time, labels, episode_type)
        if len(epochs) > 0:
            label = fp.overlay_episodes(epochs, episode_type, ax)
            vspans.append([label, episode_type])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def highlight_classifier(time: np.array, tf_array, ax=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []

    sections = np.where(tf_array is True)

    for episode_type in episodes_to_plot:

        _, epochs = find_episodes(time, labels, episode_type)
        if len(epochs) > 0:
            label = fp.overlay_episodes(epochs, episode_type, ax)
            vspans.append([label, episode_type])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax

def main(time: np.array, f_trace: np.array, labels_df: pd.DataFrame):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(30, 15), sharex=False)

    # Get the dF/F plot
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # ax1.xaxis.label.set_visible(False))
    _ = highlight_episodes(time, labels_df, plot_key="all_behaviors", ax=ax1)
    ax1.axhline(0, ls='--', c='gray')

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax2)
    _ = highlight_episodes(time, labels_df, plot_key="all_zones", ax=ax2)
    #ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax2.set_xlabel('Time')

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax3)
    _ = highlight_episodes(time, labels_df, plot_key="all_zones", ax=ax3)
    # ax2.axhline(2, ls='--', c='gray')
    ax3.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax3.set_xlabel('Time')

    return fig


if __name__ == "__main__":
    "Code to test that the plotting works"

    mouse_ID = 4
    day = 1
    id = "{}.{}".format(mouse_ID, day)

    behavior_dir = paths.behavior_scoring_directory
    dff_dir = paths.processed_data_directory
    save_directory = paths.figure_directory
    f_io.check_dir_exists(save_directory)

    behavior_labels = f_io.load_behavior_labels(id)
    data = f_io.load_preprocessed_data(id)
    # behavior_labels = f_io.load_behavior_labels(id, base_directory=row['Behavior Labelling'])
    # data = f_io.load_preprocessed_data(id, base_directory=row['Preprocessed Data'])

    fig = main(data['time'], data['zscore'], behavior_labels)
    plt.suptitle(" ".join((str(data['ani_id']), 'Z-Score DFF', 'behavior segmentation')))
    # plt.savefig(join(save_directory, " ".join((str(Ani_ID), 'Z-Score DFF', 'behavior segmentation')) + ".png"))
    plt.show()
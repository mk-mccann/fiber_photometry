import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join

# %matplotlib qt

import paths
import functions_plotting as fp
import functions_io as f_io
from functions_utils import find_episodes


def highlight_manual_episodes(time: np.array, labels: pd.DataFrame, plot_key="all_behaviors", ax=None):
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
    elif plot_key == "ALL":
        episodes_to_plot = [" ".join(col.split(" ")[0:-1]) for col in labels.columns]
    else:
        episodes_to_plot = plot_key

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []
    for episode_type in episodes_to_plot:

        _, epochs = find_episodes(time, labels, episode_type)
        if len(epochs) > 0:
            label = fp.overlay_manual_episodes(epochs, episode_type, ax)
            vspans.append([label, episode_type])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def highlight_glm_episodes(time: np.array, glm_labels: pd.DataFrame, ax=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []
    for col in glm_labels.columns:

        label = fp.overlay_glm_epidsodes(fp.mpl_datetime_from_seconds(time), glm_labels[col].to_numpy(), col, ax)
        vspans.append([label, col])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def main(time: np.array, f_trace: np.array, labels_df: pd.DataFrame, glm_labels: pd.DataFrame):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(30, 15), sharex=False)

    # Get the dF/F plot
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # ax1.xaxis.label.set_visible(False))
    _ = highlight_manual_episodes(time, labels_df, plot_key="all_behaviors", ax=ax1)
    ax1.axhline(0, ls='--', c='gray')

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax2)
    _ = highlight_manual_episodes(time, labels_df, plot_key="all_zones", ax=ax2)
    #ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax2.set_xlabel('Time')

    # Add a subplot containing the GLM
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax3)
    _ = highlight_glm_episodes(time, glm_labels, ax=ax3)
    ax3.set_xlabel('Time')

    return fig


def extract_glm_predictions(glm_data, mouse_id, day):
    animal_exp = glm_data.loc[(glm_data['animal'] == mouse_id) & (glm_data['day'] == day)]
    animal_exp.sort_values(['index'], inplace=True)
    return animal_exp


if __name__ == "__main__":
    "Code to test that the plotting works"

    mouse_ID = 4
    day = 1
    id = "{}.{}".format(mouse_ID, day)
    glm_prediction_key = 'Eating'

    behavior_dir = paths.behavior_scoring_directory
    dff_dir = paths.processed_data_directory
    model_dir = paths.modeling_data_directory
    save_directory = paths.figure_directory
    f_io.check_dir_exists(save_directory)

    manual_behavior_labels = f_io.load_behavior_labels(id)
    data = f_io.load_preprocessed_data(id)
    glm = f_io.load_glm_h5('classifier_df.h5')
    exp_predict = extract_glm_predictions(glm, mouse_ID, day)
    # behavior_labels = f_io.load_behavior_labels(id, base_directory=row['Behavior Labelling'])
    # data = f_io.load_preprocessed_data(id, base_directory=row['Preprocessed Data'])

    fig = main(data['time'], data['zscore'], manual_behavior_labels, exp_predict)
    plt.suptitle(" ".join((id, 'Z-Score DFF', 'behavior segmentation with GLM predictions')))
    plt.savefig(join(save_directory, " ".join((str(id), 'Z-Score DFF', 'behavior segmentation+GLM')) + ".png"))
    plt.show()

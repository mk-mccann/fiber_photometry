import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

import paths
import fp.visualization as viz
import fp.io as f_io


def highlight_glm_episodes(time: np.array, glm_predictions: pd.DataFrame, glm_keys, ax=None):
    """Overlay a fluorescence trace plot with the windows where the GLM predicts the scoring type of interest.

    Parameters
    ----------
    time : np.array
        Time series of the
    glm_predictions : pd.DataFram
    glm_keys : iterable object of str
    ax : matplotlib figure axes

    Returns
    -------
        matplotlib figure axes

    """

    if ax is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 15))

    # Create the highlighted episodes
    vspans = []
    for col, key in zip(glm_predictions.columns, glm_keys):

        label = viz.color_overlay(viz.mpl_datetime_from_seconds(time), glm_predictions[col].to_numpy(), key, ax)
        vspans.append([label, key])

    vspans = np.array(vspans)
    ax.legend(vspans[:, 0], vspans[:, 1], loc="upper right")

    return ax


def main(data_df: pd.DataFrame, glm_predictions: pd.DataFrame, glm_keys):
    """

    Parameters
    ----------
    data_df : pd.DataFrame
    glm_predictions : pd.DataFrame
    glm_keys : iterable object of strings

    Returns
    -------

    """

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(30, 15), sharex=False)

    # Plot the dF/F with manual behavior labels
    viz.plot_fluorescence_min_sec(data_df['time'], data_df['zscore'], ax=ax1)
    found_behaviors = np.unique(data_df['behavior'][data_df['behavior'] != ''])
    _ = viz.highlight_episodes(data_df, 'behavior', found_behaviors, ax=ax1)
    ax1.axhline(0, ls='--', c='gray')
    ax1.set_ylabel('Z-dF/F')
    ax1.set_title('Manual Behavior Labels')

    # Plot the dF/F with manual zone occupancy labels
    viz.plot_fluorescence_min_sec(data_df['time'], data_df['zscore'], ax=ax2)
    found_zones = np.unique(data_df['zone'][data_df['zone'] != ''])
    _ = viz.highlight_episodes(data_df, 'zone', found_zones, ax=ax2)
    #ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Z-dF/F')
    ax2.set_title('Manual Zone Occupancy')

    # Plot the dF/F with GLM predictions
    viz.plot_fluorescence_min_sec(data_df['time'], data_df['zscore'], ax=ax3)
    _ = highlight_glm_episodes(time, glm_predictions, glm_keys, ax=ax3)
    ax2.axhline(0, ls='--', c='gray')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Z-dF/F')
    ax3.set_title('GLM Prediction: ' + " ".join(*glm_keys))

    return fig


def extract_glm_predictions(glm_data, mouse_id, day):
    predict_cols = [col for col in glm_data.columns if 'prediction' in col]

    animal_exp = glm_data.loc[(glm_data['animal'] == mouse_id) & (glm_data['day'] == day)]
    animal_exp.sort_values(['index'], inplace=True)

    predictions = animal_exp[predict_cols]

    return animal_exp['ts'].to_numpy(), animal_exp['zscore'].to_numpy(), predictions


if __name__ == "__main__":

    mouse = 4
    day = 1
    glm_prediction_keys = ['Eating']

    f_io.check_dir_exists(paths.figure_directory)

    manual_behavior_labels = f_io.load_behavior_labels(mouse, day)
    data = f_io.load_preprocessed_data(mouse, day)
    glm = f_io.load_glm_h5('classifier_df.h5')

    time, zscore, exp_predict = extract_glm_predictions(glm, mouse, day)

    fig = main(data, exp_predict, glm_prediction_keys)
    plt.suptitle(" ".join(('Animal {} Day {}'.format(mouse, day), 'Z-dF/F', 'behavior segmentation + GLM predictions')))
    plt.savefig(join(paths.figure_directory, "_".join(('animal{}_day{}'.format(mouse, day), 'zdff', 'behav_seg_GLM')) + ".png"))
    plt.show()

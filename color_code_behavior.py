import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

import paths
import functions_plotting as fp
import functions_io as f_io
import functions_utils as fu


def main(data: pd.DataFrame):
    time = data['time']
    f_trace = data['zscore']

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 15), sharex=False)

    # Get the dF/F plot
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # ax1.xaxis.label.set_visible(False))
    found_behaviors = np.unique(data['behavior'][data['behavior'] != ''])
    _ = fp.highlight_episodes(data, 'behavior', found_behaviors, ax=ax1)
    ax1.axhline(0, ls='--', c='gray')

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax2)
    found_zones = np.unique(data['zone'][data['zone'] != ''])
    _ = fp.highlight_episodes(data, 'zone', found_zones, ax=ax2)
    # ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax2.set_xlabel('Time')

    return fig


if __name__ == "__main__":

    mouse = 5
    day = 1

    # Check that the figure directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Load the preprocesed data
    data = f_io.load_preprocessed_data(mouse, day)

    # Test if the preprocessed data has the behavior labels
    if 'behavior' in data.columns:
        fig = main(data)
    else:
        warnings.warn('Behavior labelling not present in DataFrame. Trying to load now.')
        try:
            behavior_labels = f_io.load_behavior_labels(mouse, day)
            behavior_bouts, zone_bouts = fu.find_zone_and_behavior_episodes(data, behavior_labels)
            data = fu.add_episode_data(data, behavior_bouts, zone_bouts)
            fig = main(data)
        except FileNotFoundError as f:
            print("Manual scoring needs to be done for this experiment.")

    plt.suptitle(" ".join(('Animal {} Day {}'.format(mouse, day), 'Z-dF/F', 'behavior segmentation')))
    # plt.savefig(join(save_directory, "_".join(('animal{}_day{}'.format(mouse, day), 'zdff', 'behavior_seg')) + ".png"))
    plt.show()

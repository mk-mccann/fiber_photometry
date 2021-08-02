import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

import paths
import functions_plotting as fp
import functions_io as f_io
import functions_utils as fu


def main(data: pd.DataFrame, channel_key=None):
    """
    Plots a 2-panel figure for the z-scored dF/F trace for a given recording site
    and overlays the manually-scored behavior and zone-occupation periods. Time is
    scales to HH:mm format for readability.

    :param data: A pandas dataframe containing fluorescence traces and behavior labels
    :param channel_key: An optional argument for specifying which fluorescence channel to use.
                        Only used in dual-fiber recordings.

    Example usage for a single fiber recording:
    fig = main(data)

    Example usage for dual fiber recording:
    fig = main(data, 'anterior')
    fig = main(data, 'posterior')
    """

    # Select the time data
    time = data['time']

    # Select the fluorescence trace of interest. It's the z-scored dF/F by default,
    # but could be changed to 'dff' in the lines below for the normal df?f trace.
    # One could even try just 'gcamp' or 'auto' to visualize those channels.
    if channel_key is None:
        f_trace = data['zscore']
    else:
        f_trace = data['zscore_' + channel_key]

    # Generate a figure with 2 panels in a single column
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=False)

    # Get the dF/F plot and highlight the times performing labeled behaviors
    # Plot the fluorescence trace on the first panel (ax1)
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # ax1.xaxis.label.set_visible(False))
    # Extract the labeled behaviors
    found_behaviors = np.unique(data['behavior'][data['behavior'] != ''])
    # Highlight the labelled behaviors on the first panel.
    _ = fp.highlight_episodes(data, 'behavior', found_behaviors, ax=ax1)
    # Draw a dashed line at y=0
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

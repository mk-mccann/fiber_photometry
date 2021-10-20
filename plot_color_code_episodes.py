import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from warnings import warn

import paths
import functions_plotting as fp
import functions_io as f_io


def main(data_df, f_trace='zscore', channel_key=None):
    """Plots a 2-panel figure for the z-scored dF/F trace for a given recording site and overlays the manually-scored
    behavior and zone-occupation periods. Time is scaled to HH:mm format for readability.

    Example usage for a single fiber recording:
    fig = main(data)

    Single fiber with the dF/F instead of Z-score
    fig = main(data, f_trace='dff')

    Example usage for dual fiber recording plotting the Z-score:
    fig = main(data, channel_key='anterior')
    fig = main(data, channel_key='posterior')

    Parameters
    ----------
    data_df : pd.DataFrame
        Fluorescence traces and scoring for an experiment
    f_trace : str, default='zscore'
        The fluorescence trace to be plotted. Options are ['auto', 'gcamp', 'dff', 'zscore'].
    channel_key : str, optional
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default is None for single-fiber recordings.

    Returns
    -------
    fig : matplotlib figure

    """

    # Select the time data
    time = data_df['time'].to_numpy()

    # Select the fluorescence trace of interest. It's the z-scored dF/F by default,
    # but could be changed to 'dff' in the lines below for the normal dF/F trace.
    # One could even try just 'gcamp' or 'auto' to visualize those channels.
    if channel_key is None:
        f_trace = data_df[f_trace].to_numpy()
    else:
        f_trace = data_df['_'.join([f_trace, channel_key])].to_numpy()

    # Generate a figure with 2 panels in a single column
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=True)

    # Get the dF/F plot and highlight the times performing labeled behaviors
    # Plot the fluorescence trace on the first panel (ax1)
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # Extract the labeled behaviors
    found_behaviors = np.unique(data_df['behavior'][data_df['behavior'] != ''])
    # Highlight the labelled behaviors on the first panel.
    _ = fp.highlight_episodes(data_df, 'behavior', found_behaviors, ax=ax1)
    # Draw a dashed line at y=0
    ax1.axhline(0, ls='--', c='gray')
    ax1.set_ylabel('Z-dF/F')

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax2)
    found_zones = np.unique(data_df['zone'][data_df['zone'] != ''])
    _ = fp.highlight_episodes(data_df, 'zone', found_zones, ax=ax2)
    # ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    # ax2.axhline(-2, ls='--', c='gray')
    ax2.set_ylabel('Z-dF/F')
    ax2.set_xlabel('Time')

    return fig


if __name__ == "__main__":

    animal = 4
    day = 1

    # Check that the figure directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Load the preprocessed data
    data = f_io.load_preprocessed_data(animal, day)

    # Some error catching - if the behavior data is not in the df, raise an error and quit
    try:
        data = f_io.check_preprocessed_df_for_scoring(data, animal, day)
        fig = main(data, channel_key='posterior')

        plt.suptitle(" ".join(('Animal {} Day {}'.format(animal, day), 'Z-dF/F', 'behavior segmentation')))
        plt.savefig(join(paths.figure_directory, "_".join(('animal{}_day{}'.format(animal, day), 'zdff', 'behavior_seg')) + ".png"))
        plt.show()
        
    except FileNotFoundError as err:
        message = "Manual scoring needs to be done for this experiment: Animal {} Day {}.".format(animal, day)
        warn(message)
        



import matplotlib.pyplot as plt
from os.path import join
from warnings import warn

import paths
import functions_plotting as fp
import functions_io as f_io
from functions_utils import check_if_dual_channel_recording


def plot_color_code_episodes(data_df, f_trace='zscore_Lerner', is_dual_channel=False):
    """Plots a figure for the selected fluorescence trace for a given recording and overlays the manually-scored
    behavior and zone-occupation periods. Time is scaled to HH:mm format for readability. If a single-fiber recording,
    produces a 2x1 figure, if a dual-fiber recording, produces a 2x2 figure

    Example usage for a single fiber recording:
    fig, title = plot_color_code_episodes(data)

    Single fiber with the dF/F instead of Lerner Z-score
    fig, title = plot_color_code_episodes(data, f_trace='dff')

    Dual fiber recording with just GCaMP
    fig, title = plot_color_code_episodes(data, f_trace='gcamp', is_dual_channel=True)

    Parameters
    ----------
    data_df : pd.DataFrame
        Fluorescence traces and scoring for an experiment
    f_trace : str, default='zscore_Lerner'
        The fluorescence trace to be plotted.
        Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner].
    is_dual_channel : bool, default=False
        Default is False for single-fiber recordings.

    Returns
    -------
    fig : matplotlib figure
    title: str
        Title of the figure generated from the input conditions

    """

    # Extract the labeled behaviors and zones
    labeled_episodes = data_df.columns[data_df.columns.get_loc('zone') + 1:]
    found_behaviors = [ep for ep in labeled_episodes if 'Zone' not in ep]
    found_zones = [ep for ep in labeled_episodes if 'Zone' in ep]

    # Build the title out of the data present
    title = ' '.join(('Animal {} Day {}', f_trace, 'Behavior Segmentation')).title()

    # Handle the single channel recordings
    if not is_dual_channel:

        fluor_data = data_df[f_trace].to_numpy()

        # Generate a figure with 2 panels in a single column
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=True)

        # Get the dF/F plot and highlight the times performing labeled behaviors
        # Plot the fluorescence trace on the first panel (ax1)
        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data, ax=ax1)
        # Highlight the labelled behaviors on the first panel.
        _ = fp.highlight_episodes(data_df, found_behaviors, ax=ax1)
        # Draw a dashed line at y=0
        ax1.axhline(0, ls='--', c='gray')
        # Add the correct axis labels
        ax1.set_ylabel(fp.fluorescence_axis_labels[f_trace])

        # Add a subplot containing the times in a certain zone
        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data, ax=ax2)
        _ = fp.highlight_episodes(data_df, found_zones, ax=ax2)
        ax2.axhline(0, ls='--', c='gray')
        ax2.set_ylabel(fp.fluorescence_axis_labels[f_trace])
        ax2.set_xlabel('Time')

    # Handle the dual channel recordings
    else:

        fluor_data_anterior = data['_'.join([f_trace, 'anterior'])].to_numpy()
        fluor_data_posterior = data['_'.join([f_trace, 'posterior'])].to_numpy()

        # Generate a figure with 2 panels in a single column
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 15), sharex=True)

        # -- Handle the anterior channel

        # Get the dF/F plot and highlight the times performing labeled behaviors
        # Plot the fluorescence trace on the first panel (axes[0])
        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data_anterior, ax=axes[0][0])
        # Highlight the labelled behaviors on the first panel.
        _ = fp.highlight_episodes(data_df, found_behaviors, ax=axes[0][0])
        # Draw a dashed line at y=0
        axes[0][0].axhline(0, ls='--', c='gray')
        axes[0][0].set_ylabel(fp.fluorescence_axis_labels[f_trace])
        axes[0][0].set_title('Anterior')

        # Add a subplot containing the times in a certain zone
        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data_anterior, ax=axes[1][0])
        _ = fp.highlight_episodes(data_df, found_zones, ax=axes[1][0])
        axes[1][0].axhline(0, ls='--', c='gray')
        axes[1][0].set_ylabel(fp.fluorescence_axis_labels[f_trace])
        axes[1][0].set_xlabel('Time')

        # -- Handle the posterior channel
        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data_posterior, ax=axes[0][1])
        _ = fp.highlight_episodes(data_df, found_behaviors, ax=axes[0][1])
        axes[0][1].axhline(0, ls='--', c='gray')
        axes[0][1].set_ylabel(fp.fluorescence_axis_labels[f_trace])
        axes[0][1].set_title('Posterior')

        fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data_posterior, ax=axes[1][1])
        _ = fp.highlight_episodes(data_df, found_zones, ax=axes[1][1])
        axes[1][1].axhline(0, ls='--', c='gray')
        axes[1][1].set_ylabel(fp.fluorescence_axis_labels[f_trace])
        axes[1][1].set_xlabel('Time')

    return fig, title


if __name__ == "__main__":

    animal = 1
    day = 2

    # Check that the figure directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Load the preprocessed data
    data = f_io.load_preprocessed_data(animal, day)

    # Which fluorescence trace do you want to plot?
    # Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner]
    f_trace = 'zscore_Lerner'

    # Some error catching - if the behavior data is not in the df, raise an error and quit
    try:
        data = f_io.check_preprocessed_df_for_scoring(data, animal, day)
        is_DC = check_if_dual_channel_recording(data)
        fig, title = plot_color_code_episodes(data, f_trace=f_trace, is_dual_channel=is_DC)

        plt.suptitle(title.format(animal, day))
        plt.tight_layout()
        plt.savefig(join(paths.figure_directory, title.format(animal, day).replace(' ', '_').lower() + ".png"))

        # Uncomment here to show the plot
        # plt.show()

        print('Done!')
        
    except FileNotFoundError as err:
        message = "Manual scoring needs to be done for this experiment: Animal {} Day {}.".format(animal, day)
        warn(message)

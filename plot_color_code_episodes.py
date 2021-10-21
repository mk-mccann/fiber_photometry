import matplotlib.pyplot as plt
from os.path import join
from warnings import warn

import paths
import functions_plotting as fp
import functions_io as f_io
from functions_utils import check_if_dual_channel_recording


def plot_color_code_episodes(data_df, f_trace='zscore_Lerner', channel_key=None):
    """Plots a figure for the selected fluorescence trace for a given recording and overlays the manually-scored
    behavior and zone-occupation periods. Time is scaled to HH:mm format for readability. If a single-fiber recording,
    produces a 2x1 figure, if a dual-fiber recording, produces a 2x2 figure

    Example usage for a single fiber recording:
    fig, title = plot_color_code_episodes(data)

    Single fiber with the dF/F instead of Lerner Z-score
    fig, title = plot_color_code_episodes(data, f_trace='dff')

    Dual fiber recording with just GCaMP
    fig1, title1 = plot_color_code_episodes(data, f_trace='gcamp', channel_key='anterior')
    fig2, title2 = plot_color_code_episodes(data, f_trace='gcamp', channel_key='posterior')

    Parameters
    ----------
    data_df : pd.DataFrame
        Fluorescence traces and scoring for an experiment
    f_trace : str, default='zscore_Lerner'
        The fluorescence trace to be plotted.
        Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner].
    channel_key : str, optional, default=None
        Fluorescence channel to use. Only used in dual-fiber recordings. Options are ['anterior', 'posterior'].
        Default=None for single-fiber recordings.

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

    if channel_key is None:
        f_trace_key = f_trace
    else:
        f_trace_key = '_'.join([f_trace, channel_key])

    # Build the title out of the data present
    title = ' '.join(('Animal {} Day {}', f_trace_key, 'Behavior Segmentation')).replace('_', ' ').title()

    fluor_data = data_df[f_trace_key].to_numpy()

    # Generate a figure with 2 panels in a single column
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=True, sharey=True)

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

    return fig, title


if __name__ == "__main__":

    animal = 4
    day = 1

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

        # Check if this is a dual-fiber experiment
        is_DC = check_if_dual_channel_recording(data)

        # Plot peri-event time histogram for the individual behaviors (as selected in 'episodes_to_analyze')
        if is_DC:
            channels = ['anterior', 'posterior']
            for channel in channels:
                fig, title = plot_color_code_episodes(data, f_trace=f_trace, channel_key=channel)
                plt.suptitle(title.format(animal, day))
                plt.tight_layout()
                plt.savefig(join(paths.figure_directory, title.format(animal, day).replace(' ', '_').lower() + ".png"))

        # Uncomment here to show the plot
        # plt.show()

        print('Done!')
        
    except FileNotFoundError as err:
        message = "Manual scoring needs to be done for this experiment: Animal {} Day {}.".format(animal, day)
        warn(message)

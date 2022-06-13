import matplotlib.pyplot as plt
from os.path import join
from warnings import warn
from scipy.ndimage import percentile_filter, gaussian_filter1d


import paths
import functions_plotting as fp
import functions_io as f_io
from functions_utils import check_if_dual_channel_recording


def plot_behaviors_on_fluo_trace(data_df, f_trace='zscore_Lerner', behaviors_to_plot='ALL', channel_key=None):
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
    if behaviors_to_plot == 'ALL':
        found_behaviors = [ep for ep in labeled_episodes if 'Zone' not in ep]
    else:
        found_behaviors = behaviors_to_plot

    if channel_key is None:
        f_trace_key = f_trace
    else:
        f_trace_key = '_'.join([f_trace, channel_key])

    # Build the title out of the data present
    title = ' '.join(('Animal {} Day {}', f_trace_key, 'Behavior Segmentation')).replace('_', ' ').title()

    fluor_data = data_df[f_trace_key].to_numpy()

    # Smooth this even more
    fluor_data_smooth = gaussian_filter1d(fluor_data, 6)

    # Generate a figure
    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()

    # Get the dF/F plot and highlight the times performing labeled behaviors
    # Plot the fluorescence trace on the first panel (ax1)
    fp.plot_fluorescence_min_sec(data_df.time.to_numpy(), fluor_data_smooth, ax=ax)
    # Highlight the labelled behaviors on the first panel.
    _ = fp.highlight_episodes(data_df, found_behaviors, ax=ax)
    # Draw a dashed line at y=0
    ax.axhline(0, ls='--', c='gray', alpha=0.2)
    # Add the correct axis labels
    ax.set_ylabel(fp.fluorescence_axis_labels[f_trace])
    ax.set_xlabel('Time')
    # Remove top and right axis
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    return fig, title


if __name__ == "__main__":

    animal = 3
    day = 1

    # Check that the figure directory exists
    f_io.check_dir_exists(paths.figure_directory)

    # Load the preprocessed data
    data = f_io.load_preprocessed_data(animal, day)

    # Which fluorescence trace do you want to plot?
    # Options are ['auto_raw', 'gcamp_raw', 'auto', 'gcamp', 'dff', 'dff_Lerner', 'zscore', 'zscore_Lerner]
    f_trace = 'zscore_Lerner'

    # -- Which behavior(s) do you want to look at?
    # If set to 'ALL', plots all behaviors
    # Otherwise, put in a list like ['Eating'] or ['Eating', 'Grooming', 'Marble Zone', ...]
    # This is true for single behaviors also!
    behaviors_to_plot = ['Eating Window', 'Grooming', 'Transfer', 'Social Interaction']

    # Some error catching - if the behavior data is not in the df, raise an error and quit
    try:
        data = f_io.check_preprocessed_df_for_scoring(data, animal, day)

        # Check if this is a dual-fiber experiment
        is_DC = check_if_dual_channel_recording(data)

        # Plot peri-event time histogram for the individual behaviors (as selected in 'episodes_to_analyze')
        if is_DC:
            channels = ['anterior', 'posterior']
            for channel in channels:
                fig, title = plot_behaviors_on_fluo_trace(data, f_trace=f_trace, behaviors_to_plot=behaviors_to_plot, channel_key=channel)
                plt.suptitle(title.format(animal, day))
                plt.tight_layout()
                plt.savefig(join(paths.figure_directory, title.format(animal, day).replace(' ', '_').lower() + ".png"))

        else:
            fig, title = plot_behaviors_on_fluo_trace(data, f_trace=f_trace, behaviors_to_plot=behaviors_to_plot)
            plt.suptitle(title.format(animal, day))
            plt.tight_layout()
            plt.savefig(join(paths.figure_directory, title.format(animal, day).replace(' ', '_').lower() + ".png"))

        # else:
        # fig, title = plot_color_code_episodes(data, f_trace=f_trace)
        # plt.suptitle(title.format(animal, day))
        # plt.tight_layout()
        # plt.savefig(join(paths.figure_directory, title.format(animal, day).replace(' ', '_').lower() + ".png"))

        # Uncomment here to show the plot
        plt.show()

        print('Done!')

    except FileNotFoundError as err:
        message = "Manual scoring needs to be done for this experiment: Animal {} Day {}.".format(animal, day)
        warn(message)

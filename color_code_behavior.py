import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
# %matplotlib qt

import paths
import functions_plotting as fp
from functions_utils import find_nearest, get_sec_from_min_sec
import functions_io as f_io


def find_episodes(time, labels, key):

    if "Zone" in key:
        start_end = labels[[" ".join([key, "In"]), " ".join([key, "Out"])]].dropna().to_numpy()
    else:
        start_end = labels[[" ".join([key, "Start"]), " ".join([key, "End"])]].dropna().to_numpy()

    # Create a vectorized version of get_sec_from_min_sec to apply to whole arrays
    vec_get_sec_from_min_sec = np.vectorize(get_sec_from_min_sec)

    if len(start_end) > 0:

        start_end_behavior = vec_get_sec_from_min_sec(start_end)
        for episode in start_end_behavior:
            start_idx, start_time = find_nearest(time, episode[0])
            end_idx, end_time = find_nearest(time, episode[1])






def plot_episodes(time: np.array, f_trace: np.array, labels_df: pd.DataFrame, plot_key="All"):

    # For a list of all MPL colors: https://matplotlib.org/stable/gallery/color/named_colors.html
    behavior_color = {'Eating': 'cyan',
                      'Grooming': 'goldenrod',
                      'Digging': 'lime',
                      'Transfer': 'forestgreen',
                      'WSW': 'indigo',
                      'Squeezed MZ Edge': 'mediumblue',
                      'Social Interaction': 'deeppink',
                      'Ear Scratch': 'firebrick',
                      'Switch': 'sienna',
                      'Idle': 'silver',
                      'Nibbling Floor': 'thistle',
                      'Nibbling Tape': 'aquamarine',
                      'Shock': 'red',
                      }

    zone_color = {'Eating Zone': 'b',
                  'Marble Zone': 'g',
                  'Nesting Zone': 'gray',
                  }

    # Searches the dataframe for each behavior type, zone type, or other action
    behaviors = [" ".join(col.split(" ")[0:-1]) for col in labels_df.columns if "Start" in col.split(" ")[-1]]
    zones = [" ".join(col.split(" ")[0:-1]) for col in labels_df.columns if "In" in col.split(" ")[-1]]

    # If we want to plot all the behaviors and zones
    if plot_key == "All":
        bejaviors_to_plot = behaviors
    else:
        episodes_to_plot = [plot_key]

    # Make a vectorized function for converting to min_sec
    vec_get_sec_from_min_sec = np.vectorize(get_sec_from_min_sec)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 15), sharex=False)

    # Get the dF/F plot
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    # ax1.xaxis.label.set_visible(False))

    # Create the highlighted episodes of behavior
    behav_vspan = []
    for behav in episodes_to_plot:

        start_end_behavior = labels_df[[behav + " Start", behav + " End"]].dropna().to_numpy()

        if len(start_end_behavior) > 0:

            start_end_behavior = vec_get_sec_from_min_sec(start_end_behavior)
            for episode in start_end_behavior:

                start_idx, start_time = find_nearest(time, episode[0])
                end_idx, end_time = find_nearest(time, episode[1])
                b = ax1.axvspan(fp.mpl_datetime_from_seconds(start_time), fp.mpl_datetime_from_seconds(end_time),
                                facecolor=behavior_color[behav],
                                alpha=0.5)

            behav_vspan.append([b, behav])

    behav_vspan = np.array(behav_vspan)
    #ax1.axhline(2, ls='--', c='gray')
    ax1.axhline(0, ls='--', c='gray')
    #ax1.axhline(-2, ls='--', c='gray')
    ax1.legend(behav_vspan[:, 0], behav_vspan[:, 1], loc="upper right")

    #TODO scratching
    # for other in others:
    #     start_instance = labels_df[[other]].dropna().to_numpy()
    #     instance_idx, instance_time = find_nearest(time_HMS, get_mpl_datetime(start_instance))

    # Add a subplot containing the times in a certain zone
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax2)
    
    zone_vspan = []
    for zone in zones:

        start_end_zone = labels_df[[" ".join([zone, "In"]), " ".join([zone, "Out"])]].dropna().to_numpy()
        
        if len(start_end_zone) > 0:
            bool_array = np.zeros(len(time))
            start_end_zone = vec_get_sec_from_min_sec(start_end_zone)
            
            for episode in start_end_zone:
    
                start_idx, start_time = find_nearest(time, episode[0])
                end_idx, end_time = find_nearest(time, episode[1])
                bool_array[start_idx:end_idx + 1] = 1
                z = ax2.axvspan(fp.mpl_datetime_from_seconds(start_time), fp.mpl_datetime_from_seconds(end_time),
                                facecolor=zone_color[zone],
                                alpha=0.2)
    
            zone_vspan.append([z, zone])
        # ax2.plot(time_HMS, bool_array, c=zone_color[zone])

    # ax2.set_ylim([0, 1])
    #ax2.axhline(2, ls='--', c='gray')
    ax2.axhline(0, ls='--', c='gray')
    #ax2.axhline(-2, ls='--', c='gray')
    ax2.set_xlabel('Time')
    
    zone_vspan = np.array(zone_vspan)
    ax2.legend(zone_vspan[:, 0], zone_vspan[:, 1], loc="upper right")

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

    Ani_ID = data['ani_id']
    fp_times = data['time']
    auto = data['auto']
    gcamp = data['gcamp']
    dff = data['dff']
    dffzscore = data['zscore']

    fig = plot_episodes(fp_times, dffzscore, behavior_labels)
    plt.suptitle(" ".join((str(Ani_ID), 'Z-Score DFF', 'behavior segmentation')))
    plt.savefig(join(save_directory, " ".join((str(Ani_ID), 'Z-Score DFF', 'behavior segmentation')) + ".png"))            # change path for
    plt.show()


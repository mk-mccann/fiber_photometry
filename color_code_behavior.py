import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from os.path import join
# %matplotlib qt
from functions_utils import find_nearest
import functions_plotting as fp




def plot_episodes(time: np.array, f_trace: np.array, labels_df: pd.DataFrame, behavior="All"):

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
                      'Idle':'silver',
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

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 15), sharex=False)

    # Get the dF/F plot
    fp.plot_fluorescence_min_sec(time, f_trace, ax=ax1)
    #ax1.xaxis.label.set_visible(False)

    # Convert seconds into HH:MM:SS format
    time_HMS = fp.mpl_datetime_from_seconds(time)

    if behavior == "All":
        behaviors_to_plot = behaviors
    else:
        behaviors_to_plot = [behavior]

    # Create the highlighted episodes of behavior
    behav_vspan = []
    for behav in behaviors_to_plot:

        start_end_behavior = labels_df[[behav + " Start", behav + " End"]].dropna().to_numpy()

        if len(start_end_behavior) > 0:
        
            for episode in start_end_behavior:
    
                start_idx, start_time = find_nearest(time_HMS, get_mpl_datetime(episode[0]))
                end_idx, end_time = find_nearest(time_HMS, get_mpl_datetime(episode[1]))
                b = ax1.axvspan(start_time, end_time, facecolor=behavior_color[behav], alpha=0.5)

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
            bool_array = np.zeros(len(time_HMS))
            
            for episode in start_end_zone:
    
                start_idx, start_time = find_nearest(time_HMS, get_mpl_datetime(episode[0]))
                end_idx, end_time = find_nearest(time_HMS, get_mpl_datetime(episode[1]))
                bool_array[start_idx:end_idx + 1] = 1
                z = ax2.axvspan(start_time, end_time, facecolor=zone_color[zone], alpha=0.2)
    
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


    save_directory = r"J:\Alja Podgornik\FP_Alja\plots"
    behavior_dir = r"J:\Alja Podgornik\FP_Alja\Multimaze scoring"
    dff_dir = r"J:\Alja Podgornik\FP_Alja\FP_processed data"

    
    behavior_filename = r"ID{}_Day{}.xlsx".format(mouse_ID, day)    # change path
    x = pd.ExcelFile(join(behavior_dir, behavior_filename), engine='openpyxl')
    behavior_labels = pd.read_excel(x, header=0, dtype=object, engine='openpyxl')

    dff_filename = r"{}.{}.npy".format(mouse_ID, day)          # change path
    data = np.load(join(dff_dir, dff_filename), allow_pickle=True)
    data = data.item()

    Ani_ID = data['ani_id']
    fp_times = data['ts']
    auto = data['auto']
    gcamp = data['gcamp']
    dff = data['dff']
    dffzscore = data['zscore']

    fig = plot_episodes(fp_times, dffzscore, behavior_labels)
    plt.suptitle(" ".join((str(Ani_ID), 'Z-Score DFF', 'behavior segmentation')))
    plt.savefig(join(save_directory, " ".join((str(Ani_ID), 'Z-Score DFF', 'behavior segmentation')) + ".png"))            # change path for
    plt.show()


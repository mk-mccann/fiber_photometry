import os
import numpy as np
import pandas as pd
from os.path import join

from tqdm import tqdm
from joblib import Parallel, delayed

import functions_io as f_io
from functions_utils import find_episodes


def find_zone_and_behavior_episodes(data, behavior_labels):
    ts = data['time']
    behaviors = [" ".join(col.split(" ")[0:-1]) for col in behavior_labels.columns if "Start" in col.split(" ")[-1]]
    zones = [" ".join(col.split(" ")[0:-1]) for col in behavior_labels.columns if "In" in col.split(" ")[-1]]

    behav_bouts = []
    for behav in behaviors:

        behav_idxs, behav_times = find_episodes(ts, behavior_labels, behav)

        for idxs, times in zip(behav_idxs, behav_times):
            behav_bouts.append([behav, idxs[0], times[0], idxs[1], times[1]])

        # start_end_behavior = behavior_labels[[behav + " Start", behav + " End"]].dropna().to_numpy()
        #
        # if len(start_end_behavior) > 0:
        #     for episode in start_end_behavior:
        #         start_idx, start_time = find_nearest(ts, get_sec_from_min_sec(episode[0]))
        #         end_idx, end_time = find_nearest(ts, get_sec_from_min_sec(episode[1]))
        #         behav_bouts.append([behav, start_idx, start_time, end_idx, end_time])
    behav_bouts = np.array(behav_bouts)

    zone_bouts = []
    for zone in zones:
        zone_idxs, zone_times = find_episodes(ts, behavior_labels, zone)

        for idxs, times in zip(zone_idxs, zone_times):
            zone_bouts.append([zone, idxs[0], times[0], idxs[1], times[1]])
        # start_end_zone = behavior_labels[[" ".join([zone, "In"]), " ".join([zone, "Out"])]].dropna().to_numpy()
        # if len(start_end_zone) > 0:
        #     for episode in start_end_zone:
        #         start_idx, start_time = find_nearest(ts, get_sec_from_min_sec(episode[0]))
        #         end_idx, end_time = find_nearest(ts, get_sec_from_min_sec(episode[1]))
        #         zone_bouts.append([zone, start_idx, start_time, end_idx, end_time])
    zone_bouts = np.array(zone_bouts)

    return behav_bouts, zone_bouts


def add_episode_data(data, behav_bouts, zone_bouts):
    df = pd.DataFrame(data=data)
    behaviors = np.unique(behav_bouts.T[0])
    zones = np.unique(np.unique(zone_bouts.T[0]))

    for zone in zones:
        bool_array = np.array([False] * len(df))
        df[zone] = bool_array

    zones = pd.DataFrame(zone_bouts, columns=['zone', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in zones.iterrows():
        df.loc[val['start_idx']:val['end_idx'], val['zone']] = True

    for behavior in behaviors:
        bool_array = np.array([False] * len(df))
        df[behavior] = bool_array

    behaviors = pd.DataFrame(behav_bouts, columns=['behavior', 'start_idx', 'start_time', 'end_idx', 'end_time'])
    for i, val in behaviors.iterrows():
        df.loc[val['start_idx']:val['end_idx'], val['behavior']] = True

    return df


def save_as_hdf(df, subfolder_name='modeling_data'):
    subfolder_path = join(main_dir, subfolder_name)
    if not os.path.isdir(subfolder_path):
        os.mkdir(subfolder_path)

    animal_id = str(df.loc[0, 'ani_id'])
    animal, day = animal_id.split('.')
    df['animal'] = animal
    df['day'] = day
    del df['ani_id']
    df = df.astype({"animal": int, "day": int})

    df.to_hdf(join(subfolder_path, animal_id + '.h5'), key='nokey')


def perform_all_single_animal(animal_id):
    try:
        data = f_io.load_preprocessed_data(animal_id)
        behavior_labels = f_io.load_behavior_labels(animal_id)
        behav_bouts, zone_bouts = find_zone_and_behavior_episodes(data, behavior_labels)
        df = add_episode_data(data, behav_bouts, zone_bouts)
        save_as_hdf(df)
    except FileNotFoundError as err:
        print(str(err))


def aggregate(modeling_data_folder):
    accu = []
    for file in tqdm(os.listdir(modeling_data_folder)):
        if file.endswith('.h5') and not ('aggregate' in file):
            animal, day, ext = file.split('.')
            accu.append(pd.read_hdf(join(modeling_data_folder, file)))
            df = pd.concat(accu)
    return df


if __name__ == "__main__":
    main_dir = r"J:\Alja Podgornik\FP_Alja"
    behavior_dir = join(main_dir, 'Multimaze scoring')
    dff_dir = join(main_dir, 'FP_processed data')
    modeling_data_folder = join(main_dir, 'modeling_data')
    summary_file_path = r'Multimaze sheet summary.xlsx'
    all_exps = f_io.read_summary_file(summary_file_path)

    Parallel(n_jobs=32, verbose=100)(delayed(perform_all_single_animal)(animal_id)
                                     for animal_id in all_exps['Ani_ID'])

    # for animal_id in tqdm(all_exps['Ani_ID']):
    #     try:
    #         perform_all_single_animal(animal_id)
    #     except FileNotFoundError as err:
    #         print(str(err))

    accu = aggregate(modeling_data_folder)
    accu.to_hdf(join(main_dir, 'modeling_data', 'aggregated.h5'), key='nokey')

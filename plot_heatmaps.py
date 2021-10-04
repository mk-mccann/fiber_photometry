import warnings

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists

import pandas as pd

import paths
import functions_aggregation as f_aggr
from aggregate_episodes_across_experiments import create_episode_aggregate_h5


def plot_trace_heatmap(episodes, f_trace='zscore', index_key='overall_episode_number'):
    # Get individual traces
    grouped = episodes.groupby([index_key])['normalized_time', f_trace]
    times = grouped.agg(list).to_numpy()[:, 0]
    traces = grouped.agg(list).to_numpy()[:, 1]

    print('hi')


def select_analysis_window(episodes, window):

    selection = episodes[episodes['normalized_time'] <= window]
    return selection


if __name__ == "__main__":
    # Check if an aggregated episode file exists. If so, load it. If not,
    # create it and load it.
    aggregate_data_filename = 'aggregate_episodes.h5'
    aggregate_data_file = join(paths.processed_data_directory, aggregate_data_filename)

    episodes_to_analyze = ['Eating']

    try:
        aggregate_store = pd.HDFStore(aggregate_data_file)
        aggregate_keys = aggregate_store.keys()
        print('The following episodes are available to analyze: {}'.format(aggregate_keys))

        df = aggregate_store.get('eating')
        df = f_aggr.filter_episodes_for_overlap(df)
        df = f_aggr.filter_episodes_by_duration(df, 35)
        df = select_analysis_window(df, 10)
        plot_trace_heatmap(df)

        # df_filtN = filter_first_n_episodes(df, 2)

        # grouped_episodes = df.groupby



        print('hi')


    except FileNotFoundError as e:
        print(e)
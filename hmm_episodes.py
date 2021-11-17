import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import networkx as nx

from functions_io import load_all_experiments
from functions_plotting import plot_heatmap


def fit_hmm_single_animal_day(df, animal, day, key='behavior'):
    sub_df = df.loc[(df.animal == str(animal)) & (df.day == str(day))]
    data = sub_df[key].to_list()

    collapsed_data = collapse_samples_to_states([data])
    collapsed_data = [list(filter(None, s)) for s in collapsed_data]
    flattened_data = np.concatenate(collapsed_data)

    unique_labels = np.unique(flattened_data)
    integer_data = np.array([np.argwhere(unique_labels == x)[0][0] for x in flattened_data])

    X = integer_data.reshape(-1, 1)

    return X, unique_labels


def collapse_samples_to_states(data):
    collapsed_data = []
    for row in data:
        collapsed = [x for x, y in zip(row[:-1], row[1:]) if x != y]
        collapsed.append(row[-1])
        collapsed_data.append(collapsed)
    return collapsed_data


def fit_hmm_across_animals(df, day, key='behavior'):
    sub_df = df.loc[df.day == str(day)]
    per_animal_data = sub_df.groupby(['animal'])[key].agg(list)

    # Collapse blocks of samples in the same state to a single observation and
    # remove unlabeled states
    collapsed_data = collapse_samples_to_states(per_animal_data)
    collapsed_data = [list(filter(None, s))for s in collapsed_data]
    flattened_data = np.concatenate(collapsed_data)
    lengths = [len(s) for s in collapsed_data]

    unique_labels = np.unique(flattened_data)
    integer_data = np.array([np.argwhere(unique_labels == x)[0][0] for x in flattened_data])

    X = integer_data.reshape(-1, 1)

    return X, lengths, unique_labels


if __name__ == "__main__":

    all_data = load_all_experiments()
    labeled_episodes = all_data.columns[all_data.columns.get_loc('zone') + 1:]

    # Get labeled behaviors, but exclude Eating and Transfer
    behaviors_to_exclude = ['Eating', 'Transfer', 'Switch', 'WSW', 'Ear Scratch', 'Nibbling Floor', 'Squeezed MZ Edge']
    found_behaviors = [ep for ep in labeled_episodes if 'Zone' not in ep]
    behaviors_to_use = [fb for fb in found_behaviors if fb not in behaviors_to_exclude]

    # Overwrite the behavior column with the behaviors we are interested in
    all_data['behavior'] = all_data[behaviors_to_use].fillna('').sum(axis=1)

    # X, lengths, states = fit_hmm_across_animals(all_data, 1, key='behavior')
    X, states = fit_hmm_single_animal_day(all_data, 2, 2, key='behavior')

    remodel = hmm.MultinomialHMM(n_components=len(states),
                                 n_iter=100, tol=0.01,
                                 verbose=True)

    # Z2 = remodel.fit(X, lengths)
    Z3 = remodel.fit(X)

    # Plot the transition matrix
    fig, ax = plot_heatmap(Z3.transmat_, states, states)
    plt.xticks(rotation=45)
    plt.gca().invert_yaxis()
    plt.show()

    print('hi')

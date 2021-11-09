import numpy as np
import pandas as pd
from hmmlearn import hmm

from functions_io import load_all_experiments


if __name__ == "__main__":
    animal = 2
    day = 1

    all_data = load_all_experiments()
    subset = all_data.loc[(all_data.animal == str(animal)) & (all_data.day == str(day))]

    # make the zones and behavior
    labeled_episodes = subset.columns[subset.columns.get_loc('zone') + 1:]
    found_behaviors = [ep for ep in labeled_episodes if 'Zone' not in ep]
    behaviors_to_use = [fb for fb in found_behaviors if fb != 'Eating']
    behaviors = subset[behaviors_to_use].fillna('').sum(axis=1).to_numpy()
    unique_behaviors = np.unique(behaviors)
    behaviors = np.array([np.argwhere(unique_behaviors == x)[0][0] for x in behaviors])
    behaviors = behaviors[behaviors > 0]

    # behaviors = subset.behavior.to_numpy
    zones = subset.zone.to_numpy()
    unique_zones = np.unique(zones)
    zones = np.array([np.argwhere(unique_zones == x)[0][0] for x in zones])

    remodel = hmm.GaussianHMM(n_components=len(np.unique(behaviors)), covariance_type='full',
                              n_iter=100, tol=0.01,
                              verbose=True)
    remodel.fit(behaviors.reshape(-1, 1))
    remodel.predict(behaviors.reshape(-1, 1))
    remodel.decode(behaviors.reshape(-1, 1))

    print('hi')

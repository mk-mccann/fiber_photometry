import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the data
data_path = r"J:\Alja Podgornik\FP_Alja\modeling_data\aggregated.h5"  # Path to aggregated.h5 file
data = pd.read_hdf(data_path)  # Read in the data
data = data.dropna(axis='columns')  # Drop columns containing na

# Prepare the pipeline
classifier = LogisticRegression(verbose=1, random_state=10061991, max_iter=10000, penalty='none', tol=1e-6)
logo_cv = LeaveOneGroupOut()
groups = data['animal'].to_numpy()
pipe = Pipeline([('scaler', StandardScaler()), ('lr_classifier', classifier)])

# Prepare the data
data = data.sample(frac=1)  # Shuffle the data
X = data['zscore'].to_numpy().reshape(-1, 1)
y = data['Eating'].to_numpy()

# Fit and report scores
scores = cross_val_score(estimator=pipe, X=X, y=y, groups=groups, cv=logo_cv,
                         n_jobs=logo_cv.get_n_splits(groups=groups))
print('scores: %s' % scores)
print('avg: %s, std: %s, var: %s' % (scores.mean(), scores.std(), scores.var()))

# ToDo: Check why shuffling the data drastically changes the std and variance of the scores despite LOGO CV

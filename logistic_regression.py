import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data_path = r"J:\Alja Podgornik\FP_Alja\modeling_data\aggregated.h5"  # Path to aggregated.h5 file
data = pd.read_hdf(data_path)  # Read in the data
data = data.dropna(axis='columns')  # Drop columns containing na

# # Subset the DataFrame
# subset_labels = ['animal', 'day', 'ts', 'zscore', 'Eating Zone', 'Eating']
# subset = data.loc[:, subset_labels]
#
# # Classify behavior
# choices = ['in', 'eating', 'out']
# conditions = [
#     subset['Eating Zone'].eq(True) & subset['Eating'].eq(False),
#     subset['Eating Zone'].eq(True) & subset['Eating'].eq(True),
#     subset['Eating Zone'].eq(False) & subset['Eating'].eq(False)
# ]
# subset['behavior'] = np.select(conditions, choices, default='error')
#
# print(subset.loc[subset['behavior'] == 'error'])  # ask Alja why this is happening?
#
# subset = subset[subset['behavior'] != 'error']
# subset = subset[subset['ts'] <= 1980]
#
# X = subset['zscore'].to_numpy().reshape(-1, 1)
# X = StandardScaler().fit_transform(X)
# y = subset['Eating Zone'].to_numpy()
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=10061991)
#
# lr = LogisticRegression(verbose=1, random_state=10061991, max_iter=10000, penalty='none', tol=1e-6)
# lr.fit(x_train, y_train)
#
# cm = confusion_matrix(y_test, lr.predict(x_test))
# cr = classification_report(y_test, lr.predict(x_test))
# print(cm)
# print(cr)
# print(lr.score(x_test, y_test))


classifier = LogisticRegression(verbose=1, random_state=10061991, max_iter=10000, penalty='none', tol=1e-6)
logo_cv = LeaveOneGroupOut()
groups = data['animal'].to_numpy()

data = data.sample(frac=1)  # Shuffle the data
X = data['zscore'].to_numpy().reshape(-1, 1)
X = StandardScaler().fit_transform(X)
y = data['Eating'].to_numpy()


scores = cross_val_score(estimator=classifier, X=X, y=y, groups=groups, cv=logo_cv,
                         n_jobs=logo_cv.get_n_splits(groups=groups))

print('scores: %s' % (scores))
print('avg: %s, std: %s, var: %s' % (scores.mean(), scores.std(), scores.var()))

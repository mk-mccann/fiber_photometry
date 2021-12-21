import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

# Load the data
data_path = r"J:\Alja Podgornik\FP_Alja\modeling_data\aggregated.h5"  # Path to aggregated.h5 file
data = pd.read_hdf(data_path)  # Read in the data
data = data.dropna(axis='columns')  # Drop columns containing na
data = data.sample(frac=1)  # Shuffle the data

# Prepare the pipeline
classifier = LogisticRegression(verbose=0, random_state=10061991, max_iter=10000, penalty='none', tol=1e-6)
logo_cv = LeaveOneGroupOut()
pipe = Pipeline([('scaler', StandardScaler()), ('lr_classifier', classifier)])

# Prepare the data
X = data['zscore'].to_numpy().reshape(-1, 1)
y = data['Eating'].to_numpy()
groups = data['animal'].to_numpy()

# Fit and report scores
scores = cross_val_score(estimator=pipe, X=X, y=y, groups=groups, cv=logo_cv,
                         n_jobs=logo_cv.get_n_splits(groups=groups))
print('Logistic regression scores: %s' % scores)
print('Avg: %s, Std: %s, Var: %s' % (scores.mean(), scores.std(), scores.var()))

# Logistic regression with last 60 data points

# Load the data
data_path = r"J:\Alja Podgornik\FP_Alja\modeling_data\aggregated.h5"  # Path to aggregated.h5 file
data = pd.read_hdf(data_path)  # Read in the data
data = data.dropna(axis='columns')  # Drop columns containing na
data.loc[:, 'index'] = data.index

# Prepare the pipeline
classifier = LogisticRegression(verbose=1, random_state=10061991, max_iter=10000, tol=1e-6, n_jobs=-1)
pipe = Pipeline([('scaler', StandardScaler()), ('lr_classifier', classifier)])

# Prepare the data
features = data.loc[:, ['index', 'ts', 'zscore', 'Eating', 'animal', 'day']].to_numpy()
target = data['Eating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, shuffle=True)

clf_df = pd.DataFrame(X_test, columns=['index', 'ts', 'zscore', 'Eating', 'animal', 'day'])
X_train = X_train[:, 2].reshape(-1, 1)
X_test = clf_df.loc[:, 'zscore'].to_numpy().reshape(-1, 1)

pipe.fit(X=X_train, y=y_train)

predictions = pipe.predict(X_test)
clf_df['prediction'] = predictions
clf_df.to_hdf(r"J:\Alja Podgornik\FP_Alja\modeling_data\classifier_df.h5", key='nokey')

cr = classification_report(y_true=y_test, y_pred=predictions)
score = accuracy_score(y_true=y_test, y_pred=predictions)
balanced_score = balanced_accuracy_score(y_true=y_test, y_pred=predictions, adjusted=False)
print(cr)
print('score', score)
print('balanced score', balanced_score)

# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(n_jobs=-1)
# X_scaled = StandardScaler().fit_transform(features)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

#
# Loading DB
#
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, header=None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#
# PIPELINE
#
pipe_lr = Pipeline([('scl', StandardScaler()),
                   ('pca', PCA(n_components=2)),
                   ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)

print("Точность конвеера: {}".format(pipe_lr.score(X_test, y_test)))

#
# CROSS SCORING
#
scores = cross_val_score(pipe_lr, X_train, y_train, cv = 10)
print(scores)

#
# Learning curve
#
pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty = 'l2'))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

print(train_sizes)
print(train_scores)
print(test_scores)

plt.plot(train_sizes, np.mean(train_scores, axis = 1), color='green', label='train')
plt.plot(train_sizes, np.mean(test_scores, axis = 1), color='red', label='test')
plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1) , np.mean(train_scores, axis=1)+np.std(train_scores, axis=1) , color='green', alpha=0.15)
plt.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1) , np.mean(test_scores, axis=1)+np.std(test_scores, axis=1) , color='red', alpha=0.15)
plt.legend()
plt.grid()
plt.xlabel('Размер выборки')
plt.ylabel('Точность')

plt.ylim([0.9, 1.0])
plt.show()


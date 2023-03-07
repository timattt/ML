from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import VotingClassifier

#
# LOADING DATA
#
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#
# MAJORITY VOTING
#
clf1 = LogisticRegression(penalty='l2', C=0.001)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy')
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([('sc', StandardScaler()), ('clf', clf1)])
pipe3 = Pipeline([('sc', StandardScaler()), ('clf', clf3)])

vc = VotingClassifier(estimators=[('pipe1', pipe1), ('pipe2', clf2), ('pipe3', pipe3)], voting='hard')

def test(clf, name):
    scores = cross_val_score(clf, X, y, cv=10)
    print("{} -> {:.2f}+-{:.2f}".format(name, np.mean(scores), np.std(scores)))
    
test(pipe1, "Logistic regression")
test(clf2, "Tree")
test(pipe3, "K-neighbors")
test(vc, "vote")
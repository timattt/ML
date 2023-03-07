from scipy.linalg.tests.test_fblas import accuracy
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#
# LOADING DATA
#
url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns = ['метка класса', 'алкоголь', 'яблочная кислота', 'зола',
                   'щелочность золы', 'магний', 'всего фенола', 'флаваноиды',
                   'фенолы нефлаваноидные', 'проантоцианины',
                   'интенсивность цвета', 'оттенок', 'разбавление', 'пролин']

df_wine = df_wine[df_wine['метка класса'] != 1]
y = df_wine['метка класса'].values
X = df_wine[['алкоголь', 'оттенок']].values

print(df_wine.head())

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#
# AdaBoost
#
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(estimator=tree, n_estimators=500, learning_rate=0.1)

#
# DRAW
#
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1,ncols=2, sharex='col', sharey='row', figsize=(8,3))

for idx, clf, tt in zip([0, 1], [tree, ada], ['Tree', 'Ada']):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy for {} is {:.2f}".format(tt, np.mean(cross_val_score(clf, X_test, y_test, cv = 10))))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1], c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol')
plt.show()
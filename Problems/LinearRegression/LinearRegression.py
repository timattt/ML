from scipy.linalg.tests.test_fblas import accuracy
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

#
# LOADING DATA
#
url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                   'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())

#
# SEABORN
#
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height=2.5)
plt.show()

#
# Correlation coefs
#
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='0.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()

#
# Linear regression
#
X = df[['RM']].values
y = df['MEDV'].values
lr = LinearRegression()
lr.fit(X, y)
plt.scatter(X, y, color='green')
plt.plot(X, lr.predict(X), color='red')
plt.show()

#
# RANSAC
#
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_threshold=2.0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

plt.scatter(X[inlier_mask], y[inlier_mask], color='blue')
plt.scatter(X[outlier_mask], y[outlier_mask], color='green')
plt.plot(X, ransac.predict(X), color='red')
plt.show()

#
# POLYNOMIAL
#
X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

# polynomial features
quadratic = PolynomialFeatures(degree = 2)
cubic = PolynomialFeatures(degree = 3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_tmp = np.arange(X.min(), X.max())[:, np.newaxis]# упоротая запись для создания нового измерения

# linear fit
regr.fit(X, y)
plt.plot(X_tmp, regr.predict(X_tmp), color='blue', label='linear')

# quad fit
regr.fit(X_quad, y)
plt.plot(X_tmp, regr.predict(quadratic.fit_transform(X_tmp)), color = 'green', label='quad')

# cube fit
regr.fit(X_cubic, y)
plt.plot(X_tmp, regr.predict(cubic.fit_transform(X_tmp)), color = 'orange', label='cubic')

# all
plt.scatter(X, y, color='gray')

plt.legend()
plt.show()

#
# DecisionTree
#
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sorted_args = X.flatten().argsort()

plt.scatter(X, y, color='gray')
plt.plot(X[sorted_args], tree.predict(X[sorted_args]), label='tree')

plt.legend()
plt.show()

#
# RandomizedForest
#
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = RandomForestRegressor(n_estimators=1000, criterion='squared_error')
tree.fit(X, y)
sorted_args = X.flatten().argsort()

plt.scatter(X, y, color='gray')
plt.plot(X[sorted_args], tree.predict(X[sorted_args]), label='forest')

plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Perceptrons.PerceptronRosenblatt import Perceptron as perRos
from Perceptrons.PerceptronAdaline import Perceptron as perAda

#
# DATA
#

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print(df.tail())

y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x')

plt.xlabel('чашелистик')
plt.ylabel('лепесток')

plt.show()

#
# DRAW METHODS
#

def drawPerceptronHyperplane(X, y, prc, h = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1min = X[:, 0].min() - 1
    x1max = X[:, 0].max() + 1
    x2min = X[:, 1].min() - 1
    x2max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, h), np.arange(x2min, x2max, h))
    
    yhat = prc.predict(np.array([xx1.ravel(), xx2.ravel()]))
    yhat = yhat.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, yhat, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y==cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx],label=cl)
        
    plt.xlabel('чашелистик')
    plt.ylabel('лепесток')
    plt.legend()
    plt.show()

def drawPerceptronEpochsAndErrors(pr):
    plt.plot(range(1, len(pr.errors_)+1), pr.errors_, marker='o')
    plt.xlabel('Эпохи')
    plt.ylabel('Кол ошибок')
    plt.show()

#
# TESTS
#
eta = 0.01
epochCount = 15

# Rosenblatt
pr = perRos(eta, epochCount)
pr.fit(X, y)
drawPerceptronHyperplane(X, y, pr, 0.02)
drawPerceptronEpochsAndErrors(pr)

# Adaline
# Нормировка признаков
X_std = np.copy(X)
for i in range(len(X[0])):
    X_std[:,i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

pr = perAda(eta, epochCount)
pr.fit(X_std, y)
drawPerceptronHyperplane(X_std, y, pr, 0.02)
drawPerceptronEpochsAndErrors(pr)
# 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

import Perceptron.SKPerceptronTest
import LogisticRegression.SKLogisticRegressionTest

# DRAW
def drawFeaturesPlane(Xtrain, ytrain, Xtest, ytest, prc, h = 0.02):
    X = np.vstack((Xtrain, Xtest))
    y = np.hstack((ytrain, ytest))
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1min = X[:, 0].min() - 1
    x1max = X[:, 0].max() + 1
    x2min = X[:, 1].min() - 1
    x2max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, h), np.arange(x2min, x2max, h))
    
    yhat = prc.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    yhat = yhat.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, yhat, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y==cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx],label=cl)

    plt.scatter(x=Xtest[:, 0], y=Xtest[:, 1], s=80, facecolors='none', edgecolors='gray', label='Тестовый набор')
        
    print("Верность: {}".format(accuracy_score(ytest, prc.predict(Xtest))))
        
    plt.xlabel('чашелистик')
    plt.ylabel('лепесток')
    plt.legend()
    plt.show()

# load data base
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# randomly split database
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3,random_state=0)

# scale data. Ensure матож is zero. And дисперсия equals to one.
sc = StandardScaler()
sc.fit(Xtrain)
X_train_std = sc.transform(Xtrain)
X_test_std = sc.transform(Xtest)

drawFeaturesPlane(X_train_std, ytrain, X_test_std, ytest, Perceptron.SKPerceptronTest.test(X_train_std, ytrain))
drawFeaturesPlane(X_train_std, ytrain, X_test_std, ytest, LogisticRegression.SKLogisticRegressionTest.test(X_train_std, ytrain))
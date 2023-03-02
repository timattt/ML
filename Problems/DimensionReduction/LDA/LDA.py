from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

#
# DRAW
#
def drawFeaturesPlane(Xtrain, ytrain, Xtest, ytest, prc, h = 0.02, name = None):
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
        
    plt.title(name)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.show()
    
#
# Load data
#
url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns = ['метка класса', 'алкоголь', 'яблочная кислота', 'зола',
                   'щелочность золы', 'магний', 'всего фенола', 'флаваноиды',
                   'фенолы нефлаваноидные', 'проантоцианины',
                   'интенсивность цвета', 'оттенок', 'разбавление', 'пролин']

print(df_wine.head())

#
# TRANSFORM DATA
#
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
    
#
# PCA
#
lda = LinearDiscriminantAnalysis(n_components = 2)
lr = LogisticRegression()
X_train_pca = lda.fit_transform(X_train_std, y_train)
X_test_pca = lda.transform(X_test_std)
lr.fit(X_train_pca, y_train)
drawFeaturesPlane(X_train_pca, y_train, X_test_pca, y_test, lr)



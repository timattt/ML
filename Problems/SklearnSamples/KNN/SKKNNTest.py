from sklearn.neighbors import KNeighborsClassifier

def test(X, y):
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X, y)
    return knn
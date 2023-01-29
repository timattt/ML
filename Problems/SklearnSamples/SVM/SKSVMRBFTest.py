from sklearn.svm import SVC

def test(X, y):
    svm = SVC(kernel='rbf', gamma = 0.2, C=10.0)
    svm.fit(X, y)
    return svm
    
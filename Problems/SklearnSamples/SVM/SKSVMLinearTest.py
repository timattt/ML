from sklearn.svm import SVC

def test(X, y):
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)
    return svm
    
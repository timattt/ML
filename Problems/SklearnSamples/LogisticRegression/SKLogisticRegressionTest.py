from sklearn.linear_model import LogisticRegression

def test(X, y):
    lr = LogisticRegression(C=1000.0)
    lr.fit(X, y)
    
    return lr
    
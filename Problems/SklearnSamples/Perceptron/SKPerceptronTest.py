from sklearn.linear_model import Perceptron

def test(X, y):
    ppn = Perceptron(max_iter=40, eta0=0.1)
    ppn.fit(X, y)
    return ppn
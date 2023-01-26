import numpy as np

class Perceptron:
    
    """
    Переменные класса:
    eta - скорость обучения
    epochCount - количество итераций обучения
    w_ - массив весов, плюс константа
    errors - кол. ошибок в каждой эпохе после обучения
    """
    
    def __init__(self, eta, epochCount):
        self.eta = eta
        self.epochCount = epochCount
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.epochCount):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                if update != 0.0:
                    errors += 1
            self.errors_.append(errors)

    def predict(self, X):
        return np.where(np.dot(self.w_[1:], X) + self.w_[0] >= 0.0, 1, -1)

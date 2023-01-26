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
            output = np.matmul(X, self.w_[1:]) + self.w_[0] * np.ones_like(y)
            errors = y - output
            self.w_[1:] += self.eta * np.dot(X.T, errors)
            self.w_[0] += self.eta * errors.sum()
            
            self.errors_.append(np.abs(errors).sum())

    def predict(self, X):
        return np.where(np.dot(self.w_[1:], X) + self.w_[0] >= 0.0, 1, -1)

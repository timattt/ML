import numpy as np
import sys
from tempfile import mkdtemp
import os.path as path
import pyprind
import time

def add_bias_unit(X, how):
    if how == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif how == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    else:
        raise AttributeError('how param must be column or row')
    return X_new

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    sg = sigmoid(z)
    return sg * (1 - sg)

def L2_reg(lambda_, w1, w2):
    return lambda_/2 * np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2)

def L1_reg(lambda_, w1, w2):
    return lambda_/2 * np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:].sum())

def encode_labels(y, k):
    result = np.zeros((k, y.shape[0]))
    for ids, val in enumerate(y):
        result[val, ids] = 1.0
    return result

class MLP:
    
    def __init__(self, n_output, n_features, n_hidden, l1=0, l2=0, epochs=500,
                 eta=0.001, alpha=0, decrease_const=0, minibatches=1):
        self.n_output_ = n_output
        self.n_features_ = n_features
        self.n_hidden_ = n_hidden
        self.l1_ = l1
        self.l2_ = l2
        self.epochs_ = epochs
        self.eta_ = eta
        self.alpha_ = alpha
        self.decrease_const_ = decrease_const
        self.minibatches_ = minibatches
        self.w1, self.w2 = self.init_weights()
        
    def init_weights(self):
        w1 = np.random.uniform(-1, 1, size=self.n_hidden_ * (self.n_features_ + 1))
        w1 = w1.reshape(self.n_hidden_, self.n_features_ + 1)
        
        w2 = np.random.uniform(-1, 1, size=self.n_output_*(self.n_hidden_ + 1))
        w2 = w2.reshape(self.n_output_, self.n_hidden_ + 1)
        
        return w1, w2

    def feed_forward(self, X, w1, w2):
        a1 = add_bias_unit(X, 'column')
        
        # LAYER I
        # input a1
        z2 = w1.dot(a1.T)
        a2 = sigmoid(z2)
        a2 = add_bias_unit(a2, 'row')
        # output a2
        
        # LAYER II
        # input a2
        z3 = w2.dot(a2)
        a3 = sigmoid(z3)
        # output a3
        
        return a1, z2, a2, z3, a3
    
    def get_grad(self, a1, a2, a3, z2, y_enc, w1, w2):
        
        sigma3 = a3 - y_enc
        z2 = add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * sigmoid_grad(z2)
        sigma2 = sigma2[1:, :]
        
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        
        grad1[:, 1:] += self.l2_ * w1[:, 1:]
        grad1[:, 1:] += self.l1_ * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2_ * w2[:, 1:]
        grad2[:, 1:] += self.l1_ * np.sign(w2[:, 1:])
        
        return grad1, grad2
    
    def get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = L1_reg(self.l1_, w1, w2)
        L2_term = L2_reg(self.l2_, w1, w2)
        cost = cost + L1_term + L2_term
        
        return cost
    
    def fit(self, X, y):
        self.cost_ = []
        
        X_data = X.copy()
        y_data = y.copy()
        y_enc = encode_labels(y, self.n_output_)
        
        delta_w1prev = np.zeros(self.w1.shape)
        delta_w2prev = np.zeros(self.w2.shape)
        
        pbar = pyprind.ProgBar(self.epochs_, monitor=True)
        
        for i in range(self.epochs_):
            #sys.stderr.write("epoch: [%d/%d]\n" % (i+1, self.epochs_))
            #sys.stderr.flush()
            
            self.eta_ /= (1 + self.decrease_const_ * i)
        
            mini = np.array_split(range(y_data.shape[0]), self.minibatches_)

            for idx in mini:
                a1, z2, a2, z3, a3 = self.feed_forward(X_data[idx], self.w1, self.w2)
            
                cost = self.get_cost(y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
            
                self.cost_.append(cost)
                
                grad1, grad2 = self.get_grad(a1, a2, a3, z2, y_enc[:, idx], self.w1, self.w2)
                
                delta_w1, delta_w2 = self.eta_ * grad1, self.eta_ * grad2
                
                self.w1 -= (delta_w1 + (self.alpha_ * delta_w1prev))
                self.w2 -= (delta_w2 + (self.alpha_ * delta_w2prev))
                
                delta_w1prev = delta_w1
                delta_w2prev = delta_w2
                
            pbar.update(force_flush=True, item_id = str("{}/{}".format(i, self.epochs_)))
        
        print(pbar)
         
    def see(self):
        print("MLP:")
        print(self.w1)
        print(self.w2)
            
    def predict(self, X):
        a1, z2, a2, z3, a3 = self.feed_forward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis = 0)
        return y_pred
    
    def save(self):
        filename = 'w1.dat'
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(self.n_hidden_, self.n_features_ + 1))
        fp[:] = self.w1[:]
        fp.flush()
        
        filename = 'w2.dat'
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(self.n_output_, self.n_hidden_ + 1))
        fp[:] = self.w2[:]
        fp.flush()
        
        filename = 'cost.dat'
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(len(self.cost_)))
        fp[:] = self.cost_[:]
        fp.flush()
        
    def load(self):
        filename = 'w1.dat'
        fp = np.memmap(filename, dtype='float32', mode='r', shape=(self.n_hidden_, self.n_features_ + 1))
        self.w1[:] = fp[:]
        fp.flush()
        
        filename = 'w2.dat'
        fp = np.memmap(filename, dtype='float32', mode='r', shape=(self.n_output_, self.n_hidden_ + 1))
        self.w2[:] = fp[:]
        fp.flush()
        
        filename = 'cost.dat'
        fp = np.memmap(filename, dtype='float32', mode='r')
        self.cost_ = fp.copy()
        fp.flush()
        
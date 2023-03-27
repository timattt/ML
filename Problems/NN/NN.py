from keras.datasets import mnist
from MLP import MLP
from sklearn.tests.test_multioutput import n_outputs
from sklearn.preprocessing.tests.test_data import n_features

import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array([X_train[i].flatten() for i in range(len(X_train))])
X_test = np.array([X_test[i].flatten() for i in range(len(X_test))])

nn = MLP(n_output=10,
         n_features=X_train.shape[1],
         n_hidden=50,
         l2=0.1,
         l1=0,
         epochs=1000,
         eta=0.001,
         alpha=0.001,
         decrease_const=0.00001,
         minibatches=50)

nn.fit(X_train, y_train)

y_test_pred = nn.predict(X_test)
print('Верность на тестовом: {}/{}'.format(np.sum(y_test_pred == y_test, axis=0), X_test.shape[0]))

plt.plot(np.arange(len(nn.cost_)), nn.cost_)
plt.show()

from sklearn.decomposition import KernelPCA
from sklearn.cluster.tests.test_k_means import n_samples
from sklearn.datasets._samples_generator import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, random_state=123)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma = 15)
X_ske = kpca.fit_transform(X, y)

plt.scatter(X_ske[y == 0, 0], X_ske[y == 0, 1], color='red', marker='^')
plt.scatter(X_ske[y == 1, 0], X_ske[y == 1, 1], color='blue', marker='o')

plt.xlabel('PC1')
plt.ylabel('PC2')

plt.show()
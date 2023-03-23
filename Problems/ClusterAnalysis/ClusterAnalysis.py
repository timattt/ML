from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster.tests.test_k_means import n_samples

#
# KMEANS
#
X, y = make_blobs(150, 2, centers = 3, cluster_std=0.5, shuffle=True)

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=0.0001)
clu = km.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.5)

for i in range(3):
    plt.scatter(np.mean(X[clu == i, 0]), np.mean(X[clu == i, 1]), marker='*', color='red')

plt.show()

#
# SIHLOUETTE
#
si = silhouette_samples(X, clu)

for i in range(3):
    sic = si[clu == i]
    sic.sort()
    plt.fill_between(np.linspace(i/3, (i+1)/3, np.size(sic)), np.zeros(sic.shape[0]), sic, label='cluster'+str(i))

plt.axhline(np.mean(si), color='black', linestyle='--')

plt.ylabel('Sihlouette')
plt.legend()
plt.show()

#
# HIERARCHY TREE
#
vars = ['X', 'Y', 'Z']
labels = ['id0', 'id1', 'id2', 'id3', 'id4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=vars, index=labels)

row_clusters = linkage(df.values, method='complete', metric='euclidean')#build tree

print(row_clusters)

dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.show()

#
# DBSCAN
#
X, y = make_moons(n_samples=200, noise=0.05)
db = DBSCAN(eps = 0.2, min_samples=5, metric='euclidean')
clu = db.fit_predict(X, y)

plt.scatter(X[clu==0, 0], X[clu ==0, 1], color='blue', label='cluster1')
plt.scatter(X[clu==1, 0], X[clu ==1, 1], color='red', label='cluster2')

plt.legend()
plt.show()
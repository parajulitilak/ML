from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from KMeans import KMeans
import pandas as pd

# centroids = [(-5, -5), (5, 5)]
# cluster_std = [1, 1]
#
# X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroids, n_features=2, random_state=2)
# km = KMeans(n_clusters=2, max_itr=100)
# km.fit_predict(X)

# plt.scatter(X[:, 1], X[:, 0])
# plt.savefig('fig_1_kmeans_self_algo.png')
# plt.show()

df = pd.read_csv('student_clustering.csv')

X = df.iloc[:, :].values

km = KMeans(n_clusters=4, max_itr=150)
y_means = km.fit_predict(X)
# print(y_means)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue')
plt.scatter(X[y_means == 2, 0], X [y_means == 2, 1], color='green')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow')
plt.savefig('max_itr = 150=.png')
plt.show()


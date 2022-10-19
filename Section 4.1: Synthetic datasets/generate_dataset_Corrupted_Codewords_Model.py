"""
Code for Section 4.1: Synthetic Datasets, part- 2a: Corrupted Codewords Model

To generate the synthetic datasets for the Corrupted Codewords Model with 5*10^5 points in {0, 1}^1000 for various values of k: {2, 10, 100, 1000, 2000}

Returns a .txt file of the required dataset
"""

from sklearn.metrics.pairwise import manhattan_distances
import numpy as np


def generation(N, D, k, epsilon = 0.2):
	"""
	Input:
	N : number of samples to generated
	D : dimension
	k : number of distinct clusters
	epsilon : cluster clarity (measure of how compact each cluster is)
	------------------------------------
	returns-
	data: (N * D) array
	centers: (k * D) array
	labels: (D*1) array
	"""

	# randomly choose k D-dimensional centers
	centers= np.random.randint(2, size= (k, D))
	labels = np.array([list(range(k))])

	data = np.array(centers, dtype = int)
	# Add N/k - 1 points into each cluster
	n_samples = int(N/k - 1)
	for i in range(k):
		rand_flips = np.random.uniform(size = (n_samples, D)) < epsilon
		samples_cluster = np.tile(centers[i], (n_samples, 1))
		np.logical_not(samples_cluster, out = samples_cluster, where = rand_flips)
		data = np.concatenate((data, samples_cluster), axis = 0)

		label_tiled = np.tile([i], (1, n_samples))
		labels = np.concatenate((labels, label_tiled), axis = 1)

	data = np.concatenate((data, labels.transpose()) , axis = 1)

	return data, centers

N, D, k = 500000, 1000, 2
data, centers = generation(N, D, k)

filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_1"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")

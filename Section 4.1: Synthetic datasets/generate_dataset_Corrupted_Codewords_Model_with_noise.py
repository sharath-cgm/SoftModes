"""
Code for Section 4.1: Synthetic Datasets, part- 2b: Corrupted Codewords Model with noise

To generate the synthetic datasets for the Corrupted Codewords Model with noise with 10^5 points in {0, 1}^500 for various values of k: {50, 200, 1000, 5000} and rho (noise): {0.1, 0.5, 0.9}

Returns a .txt file of the required dataset
"""

from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

def generation(N, D, k, noise = 0, epsilon = 0.2):
	"""
	Input:
	N : number of samples to generated
	D : dimension
	k : number of distinct clusters
	noise : fraction of samples that are noise
	epsilon : packing coefficient (measure of how compact each cluster is)
	------------------------------------
	returns-
	data: (N * D) array
	centers: (k * D) array
	labels: (D*1) array
	"""

	noise_samples = int(noise * N)
	N = int((1 - noise) * N)

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

	# noise
	if noise > 0:
		noise_data = np.random.randint(2, size= (noise_samples, D))
		noise_labels = np.random.randint(k, size = noise_samples)

		data = np.concatenate((data, noise_data), axis = 0)
		labels = np.concatenate((labels, [noise_labels]), axis = 1)

	data = np.concatenate((data, labels.transpose()) , axis = 1)


	return data, centers

N, D, k = 100000, 500, 2
noise = 0.1
data, centers = generation(N, D, k, noise)

filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_noise" + str(noise) + "_1"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")

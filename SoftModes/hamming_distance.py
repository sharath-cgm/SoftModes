"""
Pairwise Hamming distance
"""

import numpy as np
from sklearn.metrics import pairwise_distances


def hamming_distance_brute_(X, Y):
	"""
	Pairwise hamming distance between rows of matrix X and Y
	"""

	n_samples, n_features = X.shape
	n_clusters, _ = Y.shape

	# initialize
	pairwise_distance = np.ndarray( shape= (n_samples, n_clusters), dtype= int)

	for i in range(n_samples):
		for j in range(n_clusters):
			pairwise_distance[i][j] = np.count_nonzero(X[i] != Y[j])
	
	return pairwise_distance

def hamming_distance(X, Y): # parallel version of pairwise distances

	return pairwise_distances(X = X, Y = Y, metric = 'manhattan', n_jobs = 15) # n_jobs is the number of processors we would like to use.

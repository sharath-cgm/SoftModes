"""
Implementation of softmodes and k-modes algorithms for binary and larger alphabet versions.
4 functions:
1) softmodes()
2) kmodes()
3) softmodes_large_alphabet()
4) kmodes_large_alphabet()

"""

import numpy as np
from hamming_distance import hamming_distance
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score, f1_score


def softmodes(X, centers_init, max_iter, t):
	"""
	Implementation of the proposed Softmodes clustering algorithm

	Input:
	X: Data
	centers_init: Initial centers
	max_iter: Maximum no of iterations
	t: Rounding function hyperparameter

	Output: Labels for the data, centers of the clusters
	"""

	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	centers = centers_init
	center_old = centers.copy()
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	total_count = np.zeros(shape = n_clusters)


	for iter in range(max_iter):
		pairwise_distance = hamming_distance(X = X, Y = center_old)


		## label data points
		fraction_of_ones = np.zeros(shape= (n_clusters, n_features))

		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples): # parallelize
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			fraction_of_ones[labels[j]] += X[j]


		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		fraction_of_ones /= total_count.reshape(-1,1)

		rand_flips = np.random.uniform(low = 0, high = 1, size = (n_clusters, n_features))

		rounded_fraction = (fraction_of_ones**t)/ (fraction_of_ones**t + (1- fraction_of_ones)**t)

		centers = rand_flips < rounded_fraction
		centers.astype(int)

		# convergence condition
		if np.array_equal(labels, labels_old):
			break

		center_old[:] = centers
		labels_old[:] = labels

	return labels, centers

def kmodes(X, centers_init, max_iter):
	"""
	Implementation of the proposed kmodes clustering algorithm

	Input:
	X: Data
	centers_init: Initial centers
	max_iter: Maximum no of iterations

	Output: Labels for the data, centers of the clusters
	"""

	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	centers = centers_init
	center_old = centers.copy()
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	total_count = np.zeros(shape = n_clusters)


	# iterate- lloyds algorithm
	for iter in range(max_iter):
		pairwise_distance = hamming_distance(X = X, Y = center_old)

		fraction_of_ones = np.zeros(shape= (n_clusters, n_features))

		## label data points
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			fraction_of_ones[labels[j]] += X[j]

		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		fraction_of_ones /= total_count.reshape(-1,1)

		centers = fraction_of_ones > 0.5
		centers.astype(int)

		# convergence condition				
		if np.array_equal(labels, labels_old): 
			break

		center_old[:] = centers
		labels_old[:] = labels

	return labels, centers


def softmodes_large_alphabet(X, centers_init, max_iter, t, number_discrete_values_in_features):
	"""
	Implementation of the proposed Softmodes clustering algorithm for larger alphabet

	Input:
	X: Data
	centers_init: Initial centers
	max_iter: Maximum no of iterations
	t: Rounding function hyperparameter
	number_discrete_values_in_features: Number of distinct categorical values in each dimension/feature, especially for large alphabets

	Output: Labels for the data, centers of the clusters
	"""

	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	centers = centers_init
	center_old = centers.copy()
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	total_count = np.zeros(shape = n_clusters)
	count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))


	for iter in range(max_iter):
		pairwise_distance = hamming_distance(X = X, Y = center_old)

		count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))

		## label data points
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			for k in range(n_features):
				count_alphabets[labels[j]][k][X[j][k]] += 1


		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		for j in range(n_clusters):
			count_alphabets[j] /= total_count[j] # fraction

		for j in range(n_clusters):
			for k in range(n_features):

				l = number_discrete_values_in_features[k]
				z = count_alphabets[j][k]

				probability = np.zeros(shape = l)
				for i in range(l):
					probability[i] = z[i]**t

				total = np.sum(probability)
				if total != 0:
					normalize_p = probability/total
				else:
					normalize_p = np.array([1])

				centers[j][k] = np.random.choice(list(range(l)), p = normalize_p)


		# convergence condition
		if np.array_equal(labels, labels_old):
			break

		center_old[:] = centers
		labels_old[:] = labels


	return labels, centers

def kmodes_large_alphabet(X, centers_init, max_iter, number_discrete_values_in_features):
	"""
	Implementation of the proposed kmodes clustering algorithm for larger alphabet

	Input:
	X: Data
	centers_init: Initial centers
	max_iter: Maximum no of iterations
	t: Rounding function hyperparameter
	number_discrete_values_in_features: Number of distinct categorical values in each dimension/feature, especially for large alphabets

	Output: Labels for the data, centers of the clusters
	"""


	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	centers = centers_init
	center_old = centers.copy()
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	total_count = np.zeros(shape = n_clusters)
	count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))


	# iterate- lloyds algorithm
	for iter in range(max_iter):
		pairwise_distance = hamming_distance(X = X, Y = center_old)

		count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))

		## label data points
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			for k in range(n_features):
				count_alphabets[labels[j]][k][X[j][k]] += 1


		## update centers
		_, total_count = np.unique(labels, return_counts = True)

		for j in range(n_clusters):
			for k in range(n_features):
				max_count = np.max(count_alphabets[j][k])
				max_indices = np.where(count_alphabets[j][k] == max_count)[0]
				centers[j][k] = np.random.choice(max_indices)
	
		# convergence condition				
		if np.array_equal(labels, labels_old): 
			break

		center_old[:] = centers
		labels_old[:] = labels


	return labels, centers
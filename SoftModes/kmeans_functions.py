"""
k-means algorithm, k-means++ seeding and k-means class definition

"""

import random
import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score


# k-means++ seeding
def kmeans_plusplus(X, n_clusters, x_squared_norms):
	"""
 	X : data
	n_clusters : number of seeds to choose
	x_squared_norms: pre-computed X**2
 	---------

 	Returns centers
 	"""
  
	n_samples, n_features = X.shape
	centers = np.empty((n_clusters, n_features), dtype=X.dtype)
	n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = np.random.randint(0, n_samples) ##2
	centers[0] = X[center_id]

	# Initialize list of closest distances and calculate current potential
	closest_dist_sq = pairwise_distances(X = centers[0, np.newaxis], Y = X, metric = 'sqeuclidean', n_jobs = 15)
	current_pot = closest_dist_sq.sum()

	# Pick the remaining n_clusters-1 points
	for c in range(1, n_clusters):
		# Choose center candidates by sampling with probability proportional to the squared distance to the closest existing center
		probability = closest_dist_sq

		rand_vals = np.random.uniform(size=n_local_trials) * probability.sum()
		candidate_ids = np.searchsorted(np.cumsum(probability, dtype=np.float64), rand_vals) ##3
		np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)  ##4

		# Compute distances to center candidates
		distance_to_candidates = pairwise_distances(X = X[candidate_ids], Y = X, metric = 'sqeuclidean', n_jobs = 15)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		# Add best center candidate found in local tries
		centers[c] = X[best_candidate]

	return centers


# k-means algorithm
def lloyds(X, centers_init, x_squared_norms, max_iter, true_labels = None):
	"""
	Implementation of the k-means clustering algorithm

	inputs:
	X : data
	centers_init : initial centers
	x_squared_norms : Precomputed x_squared_norms
	max_iter : Maximum number of iterations, default = 300
	--------------------
	outputs:
	centers : centers of the clusters obtained
	label : Labels for the data

	"""

	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	centers = centers_init
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	pairwise_dist = np.zeros((n_samples, n_clusters), dtype=X.dtype)

	accuracy = []

	# iterate- lloyds algorithm
	for _ in range(max_iter): # convergence condition #1 : run for max_iter number of times
		pairwise_dist = pairwise_distances(X = X, Y = centers, metric = 'sqeuclidean', n_jobs = 15)

		# label data points and update centers
		count_in_clusters = np.ones(n_clusters)
		centers_new = np.zeros_like(centers)
		for j in range(n_samples):
			labels[j] = np.argmin(pairwise_dist[j])

			count_in_clusters[labels[j]] += 1
			centers_new[labels[j]] += X[j]

		# update centers
		for j in range(n_clusters):
			centers_new[j] /= count_in_clusters[j]


		centers = centers_new


		# convergence condition #2 : when labels don't change
		if np.array_equal(labels, labels_old): 
			break

		labels_old[:] = labels

		# compute accuracy
		if true_labels is not None:
			predicted_labels = labelling(labels, true_labels, n_clusters, n_samples)
			accuracy.append(accuracy_score(true_labels, predicted_labels))
	
	print(accuracy)

	return labels, centers



class Kmeans:
	"""
	methods: 
	1) __init__
	2) _init_centroids (for seeding)
	3) fit (Lloyds algo)
	"""
	def __init__(
		self,
		n_clusters = 2, # no of clusters
		init = "random", # "random" or "D1-seeding"
		n_init = 10, # no of epochs
		max_iter = 30 # maximum number of iterations in each epoch
	):
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter


	def _init_centroids(self, X, x_squared_norms, init):
		"""
		return centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters


		if init == "k-means++":
			centers = kmeans_plusplus(X, n_clusters, x_squared_norms = x_squared_norms)
		elif init == "random":
			seeds = np.random.permutation(n_samples)[:n_clusters]
			centers = X[seeds]

		return centers

	def fit(self, X, true_labels):
		"""
		X: data
		true_labels: Actual labels of data from the dataset; to get the epoch with max. accuracy against the generic max. inertia

		Returns:
		centers, labels
		"""

		n_samples, n_features = X.shape

		self.accuracy_list = []
		self.f1_list = []

		# subtract of mean of X for more accurate distance computations
		X_mean = X.mean(axis=0)
		X -= X_mean

		# precompute squared norms of data points
		x_squared_norms = np.reshape((X ** 2).sum(axis=1), (n_samples, 1))


		for i in range(self.n_init):

			# initialize centers
			centers_init = self._init_centroids(
				X, x_squared_norms = x_squared_norms, init = self.init
			)

			# run lloyd's algo
			labels, centers = lloyds(X, centers_init, x_squared_norms, max_iter = self.max_iter, true_labels = true_labels)
			self.labels_ = labels

			
			# Compute accuracy  and f1-score
			if true_labels is not None:
				predicted_labels = labelling(labels, true_labels, self.n_clusters, n_samples)
				(self.accuracy_list).append(accuracy_score(labels, true_labels))
				(self.f1_list).append(f1_score(labels, true_labels, average='macro'))

		# returning to actual domain
		X += X_mean
		self.best_centers = centers + X_mean

		return self

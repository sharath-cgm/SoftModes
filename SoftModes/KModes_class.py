"""
Defining the class KModes
"""

import numpy as np
from D1_seeding import D1_seeding
from cluster_labelling import labelling
from kmodes_softmodes_functions import softmodes, kmodes
from sklearn.metrics import accuracy_score, f1_score

class KModes:
	"""
	methods: 
	1) __init__
	2) _init_centroids (for seeding)
	3) fit (k-modes or softmodes)
	"""
	
	def __init__(
		self,
		n_clusters = 2, # no of clusters
		init = "random", # "random" or "D1-seeding"
		n_init = 10, # no of epochs
		max_iter = 30, # maximum number of iterations in each epoch
		algorithm = "softmodes", # "softmodes" or "k-modes"
		t = 2 # rounding hyperparameter
	):
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.algorithm = algorithm
		self.t = t


	def _init_centroids(self, X, init):
		"""
		returns centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters

		if init == "D1-seeding":
			centers = D1_seeding(X, self.n_clusters)
		elif init == "random":
			seeds = np.random.permutation(n_samples)[:n_clusters]
			centers = X[seeds]

		return centers


	def fit(self, X, true_labels = None, input_seed = None):
		"""
		X: data
		true_labels: Actual labels of data from the dataset; to get the epoch with max. accuracy against the generic max. inertia
		input_seed: Hard-code the initial seeding

		Returns:
		centers, labels
		"""

		n_samples, n_features = X.shape

		self.accuracy_list = []
		self.f1_list = []

		for i in range(self.n_init):

			# initialize centers
			if input_seed == None:
				centers_init = self._init_centroids(X, init = self.init)
			else:
				centers_init = input_seed
			

			# Run Clustering Algorithm
			if self.algorithm == "softmodes":
				labels, centers = softmodes(X, centers_init, self.max_iter, self.t)
			elif self.algorithm == "k-modes":
				labels, centers = kmodes(X, centers_init, self.max_iter)

			# below are the large alphabet versions of softmodes and k-modes

			elif self.algorithm == "softmodes_large_alphabet":
				# Number of Distinct Alphabets in each feature
				number_discrete_values_in_features = np.zeros(shape = n_features, dtype = int)
				for j in range(n_features):
					number_discrete_values_in_features[j] = len(np.unique(X[:, j]))

				labels, centers = softmodes(X, centers_init, self.max_iter, self.t, number_discrete_values_in_features)

			elif self.algorithm == "k-modes_large_alphabet":
				# Number of Distinct Alphabets in each feature
				number_discrete_values_in_features = np.zeros(shape = n_features, dtype = int)
				for j in range(n_features):
					number_discrete_values_in_features[j] = len(np.unique(X[:, j]))

				labels, centers = kmodes(X, centers_init, self.max_iter, number_discrete_values_in_features)


			# Compute accuracy  and f1-score
			if true_labels is not None:
				predicted_labels = labelling(labels, true_labels, self.n_clusters, n_samples)
				(self.accuracy_list).append(accuracy_score(predicted_labels, true_labels))
				(self.f1_list).append(f1_score(labels, true_labels, average='macro'))

		return self
		
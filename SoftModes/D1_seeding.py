"""
Implementation of D1-seeding, a variant of K-Means++ seeding

"""

import numpy as np
from hamming_distance import hamming_distance


def D1_seeding(X, n_clusters):

	"""
	K-Means++ seeding variant: D1- seeding

	Inputs: 
 	X : data
	n_clusters : number of seeds to choose

 	---------

 	Returns centers
 	"""

	n_samples, n_features = X.shape
	centers = np.empty((n_clusters, n_features), dtype=X.dtype)
	n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = np.random.randint(0, n_samples)
	centers[0] = X[center_id]

	# Initialize list of closest distances and calculate current potential
	closest_dist = hamming_distance(X = centers[0, np.newaxis], Y = X)
	current_pot = closest_dist.sum()

	# Pick the remaining n_clusters-1 points
	for c in range(1, n_clusters):
		# Choose center candidates by sampling with probability proportional to the hamming distance to the closest existing centers
		probability = closest_dist

		rand_vals = np.random.uniform(size=n_local_trials) * probability.sum()
		candidate_ids = np.searchsorted(np.cumsum(probability, dtype=np.float64), rand_vals)
		np.clip(candidate_ids, None, closest_dist.size - 1, out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = hamming_distance(X[candidate_ids], X)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist, distance_to_candidates, out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		# Add best center candidate found in local tries
		centers[c] = X[best_candidate]

	return centers

"""
Code for Section 4.1: Synthetic Datasets, part-1: stochastic block model

To generate the synthetic datasets for the stochastic block model with 10^4 points for various values of p and q: (0.6, 0.25), (0.45, 0.25), (0.3, 0.1), (0.3, 0)

Returns a .txt file of the required dataset
"""

import numpy as np

def generate_block(m, n, f):
	# returns a block of size m*n with f fraction of 1s

	return np.random.choice([0,1], size = (m,n), p = [1-f, f])

def generation(N, D, k = 2, p = 0.4, q = 0.1):
	"""
	This is the main function.

	Inputs:
	N: Number of points/samples
	D: Number of features
	k: Number of clusters
	p: Probability of 1s in the diagnol blocks
	q: Probability of 1s in the rest of the blocks	

	Returns: A N*(D+1) matrix where the first D columns represents the features of the datapoints and the last column represents the labels.

	"""


	data = generate_block(N, D, q)
	labels = []
	temp = 0

	block_rows, block_columns = int(N/k), int(D/k)
	for i in range(k):
		data[i*block_rows:(i+1)*block_rows, i*block_columns:(i+1)*block_columns] = generate_block(block_rows, block_columns, p)

		label_tiled = np.tile([i], (1, block_rows))
		if temp == 0:
			temp = 1
			labels = label_tiled
		else:
			labels = np.concatenate((labels, label_tiled), axis = 1)

	data = np.concatenate((data, labels.transpose()) , axis = 1)

	return data


# setting the parameters
N, D, k = 10000, 10000, 2
p, q = 0.3, 0.1
data = generation(N, D, k, p, q)

# write the dataset in the .txt file
filename = "N" + str(N) + "_D" + str(D) + "_k" + str(k) + "_1"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")

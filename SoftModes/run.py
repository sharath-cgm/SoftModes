"""
The run file of k-means (that we implemented), k-modes and softModes algorithms.
For convenience, we implemented two versions of k-modes and softModes each: 1) For binary data 2) Discrete large alphabet data (this is more general)

Run instructions:
Datasets are taken as input as N*(d+1) matrix where the first d columns are the features and the last column is the label of the sample
Place the dataset file in the same directory and change the 'dataset_name' (line 20) accordingly.
Run "python3 run.py"
"""

import numpy as np
from KModes_class import KModes
from performance_measures import accuracy_analysis, f1_analysis
import time
from kmeans_function import Kmeans
from cluster_labelling import labelling


# load dataset
dataset_name = "N100_D20_k2_1.txt"

print(dataset_name)
data = np.loadtxt(dataset_name)
data = data.astype(int)
labels = data[:, -1]
data = data[:, 0:-1]

(n_samples, n_features), n_clusters = data.shape, np.unique(labels).size
print(f"# clusters: {n_clusters}; # samples: {n_samples}; # features {n_features}")

# tuning some hyperparameters
n_init, max_iter = 3, 10
print("n_init, max_iter", n_init, max_iter)



################
# lloyds
print("\nLloyd's results:")
lloyd = Kmeans(init="k-means++", n_clusters=n_clusters, n_init= n_init, max_iter = max_iter)

start_time = time.time()
lloyd.fit(X = data.astype(float), true_labels = labels)

print("--- %s seconds ---" % (time.time() - start_time))

# Performance measures
accuracy_analysis(lloyd.accuracy_list)
f1_analysis(lloyd.f1_list)



################
# softmodes
print("\nsoftmodes results:")
softmodes = KModes(init="D1-seeding", n_clusters=n_clusters, n_init= n_init, max_iter = max_iter, algorithm = "softmodes", t = 2.5)

start_time = time.time()
softmodes.fit(X = data, true_labels = labels)

print("--- %s seconds ---" % (time.time() - start_time))

# Performance measures
accuracy_analysis(softmodes.accuracy_list)
f1_analysis(softmodes.f1_list)




################
# kmodes
print("\nk-modes results:")
kmodes = KModes(init="D1-seeding", n_clusters=n_clusters, n_init= n_init, max_iter = max_iter, algorithm = "k-modes")

start_time = time.time()

kmodes.fit(data, true_labels = labels)

print("--- %s seconds ---" % (time.time() - start_time))

# Performance measures
accuracy_analysis(kmodes.accuracy_list)
f1_analysis(kmodes.f1_list)

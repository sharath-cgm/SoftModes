"""
KMeans, Birch and Gaussian Clustering Algorithms from scikit-learn library

Place the dataset file in the same directory and change the 'dataset_name' (line 18) accordingly.
Make sure the files 'cluster_labelling.py' and 'performance_measures' are present in the same directory
"""

from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import numpy as np
from performance_measures import performance_measures, accuracy_analysis, f1_analysis
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score, f1_score

# load data
dataset_name = "N100_D20_k2_1.txt"

print(dataset_name)
dataset = np.loadtxt(dataset_name)
labels = dataset[:, -1]
labels = labels.astype(int)
data = dataset[:, 0:-1]

(n_samples, n_features), n_clusters = data.shape, np.unique(labels).size

print(f"# classes: {n_clusters}; # samples: {n_samples}; # features {n_features}")


#### BIRCH
print("Results of BIRCH\n")

best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 3

for i in range(no_epochs):
	# shuffle data
	# np.random.shuffle(dataset)
	# labels = dataset[:, -1]
	# labels = labels.astype(int)
	# data = dataset[:, 0:-1]

	# run algorithm
	brc = Birch(n_clusters = n_clusters).fit(data)

	# print(brc.labels_)

	# analysis
	predicted_labels = labelling(brc.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)



accuracy_analysis(accuracy_list)
f1_analysis(f1_list)

print("\n\n")


########### lloyds with kmeans++ seeding
print("Results of Lloyds with k-means++ seeding\n")

best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 10
for _ in range(no_epochs):

	kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=1,random_state=None)
	kmeans.fit(data)

	# analysis
	predicted_labels = labelling(kmeans.labels_, labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)
	# print(predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)


accuracy_analysis(accuracy_list)
f1_analysis(f1_list)



############# gaussian mixture
print("Results of Gaussian mixture\n")

best_accuracy, best_f1 = None, None
accuracy_list, f1_list = [], []

no_epochs= 5
for _ in range(no_epochs):
	gm = GaussianMixture(n_components = n_clusters)
	gm.fit(data)

	predicted_labels = labelling(gm.predict(data), labels, n_clusters, n_samples)

	accuracy = accuracy_score(labels, predicted_labels)

	accuracy_list.append(accuracy)

	f1 = f1_score(labels, predicted_labels, average='macro')
	f1_list.append(f1)


accuracy_analysis(accuracy_list)
f1_analysis(f1_list)




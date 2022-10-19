"""
The performance metric used in the work are Accuracy and F1-score based on the ground truth
The following code returns the Accuracy and F1-score measures: Mean, Standard Deviation and Maximum
"""

import numpy as np

def accuracy_analysis(accuracy_list):
	average = np.average(accuracy_list)
	standard_deviation = np.std(accuracy_list)
	maximum = max(accuracy_list)

	print("Accuracy Analysis: Average = ", average, " ; Standard Deviation = ", standard_deviation, "; Best Accuracy = ", maximum)

	return None

def f1_analysis(f1_list):
	average = np.average(f1_list)
	standard_deviation = np.std(f1_list)
	maximum = max(f1_list)

	print("F1-score Analysis: Average = ", average, " ; Standard Deviation = ", standard_deviation, "; Best F1_score = ", maximum)

	return None

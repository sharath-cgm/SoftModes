This is the code for the paper: 
### 'Clustering Categorical Data: Soft Rounding k-modes' with authors- Surya Teja Gavva, Karthik C.S., Sharath Punna
Link: https://arxiv.org/abs/2210.09640



## About Code:
-> Libraries used and their installation:
1) numpy: pip3 install numpy
2) scikit-learn: pip3 install -U scikit-learn
3) pyclustering: pip3 install pyclustering
4) dlib: pip3 install dlib

-> Description of Folders:
1) Other Clustering Algorithms:- Includes code of clustering algorithms used for comparison- Lloyds, BIRCH, Gaussian (from scikit-learn), ROCK (from pyclustering)
2) Section 4.1: Synthetic datasets:- Code for the generation of stochastic block model, corrupted codeword model with and without noise
3) Section 4.2: Real World Datasets:- Links of the real datasets from UCI Machine Learning Repository 
4) SoftModes:- Code for the proposed SoftModes algorithm

-> How to run code: 
SoftModes: 
1) Make sure the dataset file in the same directory and change the 'dataset_name' on line 20 of run.py
2) Make necessary changes to the parameters on the line 32, 56 of run.py
3) Code can be run by 'python3 run.py'

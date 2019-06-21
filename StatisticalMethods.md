# Statistical Methods

## Principal Component Analysis

[sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

 Input: set of possibly correlated variables used to predict a final variable

   Output: Seprated groups of those variables all linearly uncorrelated to one another

   Why: used to seperate variables into the best state to be used for a prediction

Structure of Output: an array of variables with the first object having the highest variance, each proceding object has the next highest variance while still being orthongal to the preceding 					variables

Components per object: component score(the variables values corresponding to a specific data point) and the loadings(the weigh each origional variable should be multiplied by to get the 						component score)

How to Do it:

1) spectral decompisiton of a data covariance matrix (the thing they did in the original paper that you liked)

OR

1) singular value decomposition of a design matrix(Each row represents an individual object, with the successive columns corresponding to the variables)
 
    (NOTE) if done this way you need to normalize the data, normalization of the data through mean centering and normalizing each variables variance to 1 (z scores could be useful)


## Unsupervised Hierarchical clustering

No singular command in python to preform this

Functions used:
[scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html), 
[sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)


[Useful Guide](https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/)

Following are the steps involved in agglomerative clustering:

1) At the start, treat each data point as one cluster. Therefore, the number of clusters at the start will be K, while K is an integer representing the number of data points.

2) Form a cluster by joining the two closest data points resulting in K-1 clusters.
Form more clusters by joining the two closest clusters resulting in K-2 clusters.
Repeat the above three steps until one big cluster is formed.

3) Once single cluster is formed, dendrograms are used to divide into multiple clusters depending upon the problem. We will study the concept of dendrogram in detail in an upcoming section.

There are different ways to find distance between the clusters. The distance itself can be Euclidean or Manhattan distance. Following are some of the options to measure distance between two clusters:

1) Measure the distance between the closes points of two clusters.
2) Measure the distance between the farthest points of two clusters.
3) Measure the distance between the centroids of two clusters.
4) Measure the distance between all possible combination of points between the two clusters and take the mean.

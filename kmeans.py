import numpy as np
import pandas as pd



def find_closest_centroids(X,centroids): 
    idx = np.zeros(X.shape[0],dtype = int) 
    """ calculating distance between each of the training examples 
    and each of the clusters and storing it in 'distance'
    """
    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            euclidean_norm = np.sqrt(np.sum(X[i] - centroids[j])**2)
            distance.append(euclidean_norm)
        # argmin will give the index of min value over the axis
        idx[i] = np.argmin(distance) 
    """ idx holds the index of the closest cluster
    for each training example """
    return idx


X = np.random.uniform(1,101,size = (100,2))
print(X[:5])

random_indices = np.random.choice(X.shape[0],size=3,replace=False)
random_points = X[random_indices]
""" selecting random 3 points from training example as initial centroids"""
centroids = random_points
idx = find_closest_centroids(X,centroids)
print("First 3 elements in idx : ", idx[:3])
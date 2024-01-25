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
            euclidean_norm = np.sqrt(np.sum(X[i] - centroids[j])**2,axis=1)
            distance.append(euclidean_norm)
        # argmin will give the index of min value over the axis
        idx[i] = np.argmin(distance) 
    """ idx holds the index of the closest cluster
    for each training example """
    return idx
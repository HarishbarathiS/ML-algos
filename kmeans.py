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

"""Recomputing centroids by averaging out the examples that
have been aassigned to same cluster"""
def compute_centroids(X,idx,K):
    row,col = X.shape
    """k -> No. of clusters 
       col -> point dimension """
    centroids = np.zeros((K,col))

    for cluster_num in range(K):
        points = X[idx == cluster_num]
        centroids[cluster_num] = np.mean(points,axis=0) 
        """ axis = 0 -> column-wise 
            ((All X's + All Y's)/number of training examples mapped to that cluster_num)"""
    return centroids



X = np.random.uniform(1,101,size = (100,2))
print(X[:5])

random_indices = np.random.choice(X.shape[0],size=4,replace=False)
random_points = X[random_indices]
""" selecting random 3 points from training example as initial centroids"""
initial_centroids = random_points
idx = find_closest_centroids(X,initial_centroids)
print("First 3 elements in idx : ", idx[:3])

print("Initial centroids : ", initial_centroids)
new_centroids = compute_centroids(X,idx,4)
print("New centroids : ", new_centroids)
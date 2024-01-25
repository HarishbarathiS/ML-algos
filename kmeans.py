import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def RunKMeans(X,initial_centroids,max_iters=10):
    row,col = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    prev_centroids = centroids
    idx = np.zeros(row)

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i,max_iters-1))
        idx = find_closest_centroids(X,centroids)
        centroids = compute_centroids(X,idx,K)
    plt.show()
    return centroids,idx



X = np.random.uniform(1,101,size = (100,2))
print(X[:5])

random_indices = np.random.choice(X.shape[0],size=4,replace=False)
random_points = X[random_indices]
""" selecting random 3 points from training example as initial centroids"""
initial_centroids = random_points
max_iters = 10
centroids,idx = RunKMeans(X,initial_centroids,max_iters)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data Points')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')

# Annotate centroids with cluster numbers
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f'Cluster {i+1}', fontsize=12, ha='center', va='center')

# Set plot labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-means Clustering')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
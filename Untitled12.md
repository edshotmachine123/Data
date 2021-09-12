```python
import numpy as np
import matplotlib.pyplot as plt

class kmeans:
    """Apply kmeans algorithm"""
    def __init__(self, num_clusters, max_iter=1000):
        """Initialize number of clusters"""
        
        self.num_clusters = num_clusters
        self.max_iter = max_iter
    
    def initalize_centroids(self, X):
        """Choosing k centroids randomly from data X"""
        
        idx = np.random.permutation(X.shape[0])
        centroids = X[idx[:self.num_clusters]]
        return centroids
        
    def compute_centroid(self, X, labels):
        """Modify centroids by finding mean of all k partitions"""
        
        centroids = np.zeros((self.num_clusters, X.shape[1]))
        for k in range(self.num_clusters):
            centroids[k] = np.mean(X[labels == k], axis=0)
            
        return centroids
    
    def compute_distance(self, X, centroids):
        """Computing L2 norm between datapoints and centroids"""

        distances = np.zeros((X.shape[0], self.num_clusters))
        
        for k in range(self.num_clusters):
            dist = np.linalg.norm(X - centroids[k], axis=1)
            distances[:,k] = np.square(dist)
            
        return distances
    
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)
    
    def fit(self, X):
        self.centroids = self.initalize_centroids(X)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroid(X, self.labels)
            
            if np.all(old_centroids == self.centroids):
                break
        
    def compute_sumstar(self, distances):
        """Computing sum total of all distances"""
        pass
np.random.RandomState(1234)
data = -2 * np.random.rand(1000, 2) 
data[500:] = 1 + 2 * np.random.rand(500,2)

 

plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:, 1])
plt.show()
```


![png](output_0_0.png)



```python
kmeansmodel = kmeans(num_clusters=2, max_iter=100)
kmeansmodel.fit(data)
centroids = kmeansmodel.centroids

centroids[0]



plt.figure(figsize=(6,6))
plt.scatter(data[kmeansmodel.labels == 0, 0], data[kmeansmodel.labels == 0, 1], c = 'green', label = 'Cluster 1')
plt.scatter(data[kmeansmodel.labels == 1, 0], data[kmeansmodel.labels == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c = 'red', s = 300, label = 'centroid')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
```


![png](output_1_0.png)



```python

```

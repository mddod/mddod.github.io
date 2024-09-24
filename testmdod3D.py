import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mdod

# Generate 200 three-dimensional clustered normal data 
np.random.seed(None)  
data_size = 200  
clusters = 100  
points_per_cluster = data_size // clusters
data = []
for _ in range(clusters):
    cluster_center = np.random.uniform(low=-10, high=10, size=(3,))
    cluster_data = cluster_center + np.random.randn(points_per_cluster, 3) * 0.5
    data.append(cluster_data)
data = np.vstack(data)

# Generate outlier data
num_outliers = 10  
outliers = np.random.uniform(low=-50, high=50, size=(num_outliers, 3))

# Merge normal data and outlier data
data_with_outliers = np.vstack([data, outliers])

# Use mdod library to detect outlier values
nd = 1  # value of the observation point in the new dimension
sn = 15 # number of statistics on the first few numbers in the order of scores from large to small
result = mdod.md(data_with_outliers, nd, sn)

# Extract outlier score (the first column in result)
scores = np.array([float(r[0]) for r in result])  # Extract column 1 as an outlier score

# Find the data points with the lowest score as outlier values
outliers_indices = np.argsort(scores)[:num_outliers]  # Find the indexes with the lowest scores
normal_indices = np.setdiff1d(np.arange(len(scores)), outliers_indices)  # Find the index of the normal value

# Print and visualize the data set
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_with_outliers[normal_indices, 0], 
           data_with_outliers[normal_indices, 1], 
           data_with_outliers[normal_indices, 2], 
           c='blue', label='Normal Data')
ax.scatter(data_with_outliers[outliers_indices, 0], 
           data_with_outliers[outliers_indices, 1], 
           data_with_outliers[outliers_indices, 2], 
           c='red', label='Outliers')
ax.set_title('Outliers vs Normal Data (3D)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()

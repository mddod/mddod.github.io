import numpy as np
import matplotlib.pyplot as plt
import mdod

# Generate 200 normal data distributed in clusters
np.random.seed(None)  
data_size = 200  
clusters = 100  
points_per_cluster = data_size // clusters
data = []
for _ in range(clusters):
    cluster_center = np.random.uniform(low=-10, high=10, size=(2,))
    cluster_data = cluster_center + np.random.randn(points_per_cluster, 2) * 0.5
    data.append(cluster_data)
data = np.vstack(data)

# Generate outlier data
num_outliers = 10  # number of outlier data
outliers = np.random.uniform(low=-50, high=50, size=(num_outliers, 2))

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
plt.figure(figsize=(10, 6))
plt.scatter(data_with_outliers[normal_indices, 0], data_with_outliers[normal_indices, 1], 
            c='blue', label='Normal Data')
plt.scatter(data_with_outliers[outliers_indices, 0], data_with_outliers[outliers_indices, 1], 
            c='red', label='Outliers')
plt.title('Outliers vs Normal Data (2D)')
plt.legend()
plt.show()

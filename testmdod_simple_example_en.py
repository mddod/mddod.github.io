import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from sklearn.decomposition import PCA
from mdod import MDOD  # MDOD module

# Define test parameters
contamination = 0.15  # Contamination ratio
norm_distance = 1.0  # MDOD parameter
top_n = 20  # MDOD parameter
sampling_rate = 0.05  # Sampling rate
random_state = 42  # Random seed

# Read data from file
data_gen_df = pd.read_csv('testmdod_simple_example_input_data.csv')
# Get feature columns (Feature_1, Feature_2, ...)
feature_columns = [col for col in data_gen_df.columns if col.startswith('Feature_')]
X = data_gen_df[feature_columns].values
# Get number of samples and features
n_samples = X.shape[0]
n_features = X.shape[1]
outliers_num = int(n_samples * contamination)  # Calculate number of outliers
print(f"Number of features: {n_features}")
print(f"Number of samples: {n_samples}")
print(f"Number of outliers: {outliers_num}")

# Run MDOD and measure execution time
mdod = MDOD(norm_distance=norm_distance, top_n=top_n, contamination=contamination, 
            sampling_rate=sampling_rate, random_state=random_state)
start_time = time.time()
mdod.fit(X)
mdod_time = time.time() - start_time
mdod_scores = mdod.decision_scores_
mdod_labels = mdod.labels_  # Get MDOD predicted labels

# Generate anomaly labels based on decision scores (top contamination proportion as anomalies)
anomaly_indices = np.argsort(mdod_scores)[-outliers_num:]  # Get indices of top outliers_num scores
anomaly_labels = np.zeros(n_samples)  # Initialize all as normal (0)
anomaly_labels[anomaly_indices] = 1  # Mark anomaly points as 1

# Add decision scores and anomaly labels to the original DataFrame
data_gen_df['MDOD_Score'] = mdod_scores  # Add decision scores column
data_gen_df['Anomaly_Label'] = anomaly_labels  # Add anomaly labels column

# Save the modified DataFrame to an output file
output_file = 'testmdod_simple_example_output_data.csv'
data_gen_df.to_csv(output_file, index=False)
print(f"Data saved to: {output_file}")

# Print runtime and related information
print(f"Sampling rate used: {sampling_rate}")
print(f"MDOD runtime: {mdod_time:.4f} seconds")
print("\nMDOD decision scores (top 10 - highest values):")
print(np.sort(mdod_scores)[-10:][::-1])  # Sort and take top 10 scores in descending order
print("\nMDOD decision scores (bottom 10 - lowest values):")
print(np.sort(mdod_scores)[:10])  # Sort and take bottom 10 scores

# Visualization function
def visualize_data(X, labels, title, true_labels=None):
    if n_features == 2:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Predicted')
        if true_labels is not None:
            plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='coolwarm', alpha=0.3, marker='x', label='True')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Label (0=Normal, 1=Outlier)')
        plt.legend()
        plt.show(block=True)
    elif n_features == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
        if true_labels is not None:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=true_labels, cmap='coolwarm', alpha=0.3, marker='x')
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show(block=True)
    else:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
        if true_labels is not None:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=true_labels, cmap='coolwarm', alpha=0.3, marker='x')
        plt.title(f"{title} (PCA Reduced to 2D)")
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Label (0=Normal, 1=Outlier)')
        plt.show(block=True)

# Call visualization function
visualize_data(X, mdod_labels, 'MDOD Predictions', anomaly_labels)

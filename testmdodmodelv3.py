import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from pyod.models.lof import LOF
from mdod import MDOD

# Define test parameters (customizable)
n_samples = 1000  # Total number of test points
outliers_num = 150  # Number of abnormal values
n_features = 2  # Dimension
contamination = outliers_num / n_samples  # Abnormal proportion
norm_distance = 1.0  # MDOD parameters
top_n = 20  # MDOD parameters
sampling_rate = 0.05  # MDOD parameters，0 < sampling_rate <= 1，1 For full quantity
random_state = 42  # Add a random seed for sampling to make it reproducible
LOF_n_neighbors = 100

# Generate synthetic data: normal points are in the central cluster within a limited range, while abnormal points are scattered on the periphery.
def generate_clustered_data(n_samples, n_features, mean, cov, radius_limit=1.0, max_attempts=1000):
    attempt = 0
    while attempt < max_attempts:
        X = np.random.multivariate_normal(mean, cov, size=n_samples)
        norms = np.linalg.norm(X - mean, axis=1)
        if np.all(norms <= radius_limit):
            print(f"Data generation succeeded after {attempt + 1} attempts.")
            return X
        attempt += 1
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    norms = np.linalg.norm(X - mean, axis=1)
    scale_factor = radius_limit / np.max(norms)
    X = X * scale_factor + mean
    print(f"Data generation failed after {max_attempts} attempts. Scaled data to fit within {radius_limit}.")
    return X

# Normal data: central clusters within a limited range
center_sigma = 0.3
try:
    normal = generate_clustered_data(n_samples - outliers_num, n_features, 
                                   mean=np.zeros(n_features), 
                                   cov=np.eye(n_features) * center_sigma, 
                                   radius_limit=1.0)
except ValueError as e:
    print(f"Error in normal data generation: {e}")
    exit(1)

# Abnormal data: scattered around the perimeter
r_min, r_max = 1.5, 3.0
outliers = np.random.randn(outliers_num, n_features)
norms = np.linalg.norm(outliers, axis=1, keepdims=True)
outliers = outliers / norms * np.random.uniform(r_min, r_max, size=(outliers_num, 1))

# Merge Data
X = np.vstack([normal, outliers])
y_true = np.concatenate([np.zeros(n_samples - outliers_num), np.ones(outliers_num)])

print(f"Generate data: {n_samples} Sample, {n_features} Dimension, {outliers_num} Outliers")

# Save the generated data to a local file
data_gen_df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
data_gen_df['True_Label'] = y_true
data_gen_df.to_csv('generated_data.csv', index=False)
print("Generated data has been saved to  'generated_data.csv'")

# Read data from a file
data_gen_df = pd.read_csv('generated_data.csv')
X = data_gen_df[[f"Feature_{i+1}" for i in range(n_features)]].values
y_true = data_gen_df['True_Label'].values

# Use MDOD and measure the time
mdod = MDOD(norm_distance=norm_distance, top_n=top_n, contamination=contamination, 
            sampling_rate=sampling_rate, random_state=random_state)
start_time = time.time()
mdod.fit(X)
mdod_time = time.time() - start_time
mdod_scores = mdod.decision_scores_
mdod_labels = mdod.labels_
mdod_auc = roc_auc_score(y_true, mdod_scores)

print(f"Use sampling rate: {sampling_rate}")

print("\n MDOD Decision Score (Top 10):")
print(mdod_scores[:10])
print("MDOD Predicted Labels (Number of Anomalies):", np.sum(mdod_labels))
print("MDOD AUC:", mdod_auc)
print(f"MDOD Runtime: {mdod_time:.6f} seconds")

# Calculate more metrics for MDOD
mdod_cm = confusion_matrix(y_true, mdod_labels)
mdod_precision = precision_score(y_true, mdod_labels)
mdod_recall = recall_score(y_true, mdod_labels)
mdod_f1 = f1_score(y_true, mdod_labels)

print("MDOD Confusion Matrix:")
print(mdod_cm)
print(f"MDOD Precision: {mdod_precision:.4f}")
print(f"MDOD Recall: {mdod_recall:.4f}")
print(f"MDOD F1-Score: {mdod_f1:.4f}")

# Using PYOD's LOF as a comparison
lof = LOF(n_neighbors=LOF_n_neighbors, contamination=contamination)
#lof = LOF(n_neighbors=top_n, contamination=contamination)
start_time = time.time()
lof.fit(X)
lof_time = time.time() - start_time
lof_scores = lof.decision_scores_
lof_labels = lof.labels_
lof_auc = roc_auc_score(y_true, lof_scores)

print("\nLOF Decision Score (Top 10):")
print(lof_scores[:10])
print("LOF Predicted Labels (Number of Anomalies):", np.sum(lof_labels))
print("LOF AUC:", lof_auc)
print(f"LOF Running time: {lof_time:.6f} seconds")

# Calculate more metrics for LOF
lof_cm = confusion_matrix(y_true, lof_labels)
lof_precision = precision_score(y_true, lof_labels)
lof_recall = recall_score(y_true, lof_labels)
lof_f1 = f1_score(y_true, lof_labels)

print("LOF Confusion Matrix:")
print(lof_cm)
print(f"LOF Precision: {lof_precision:.4f}")
print(f"LOF Recall: {lof_recall:.4f}")
print(f"LOF F1-Score: {lof_f1:.4f}")

# Compare
correlation, p_value = spearmanr(mdod_scores, lof_scores)
print("\nComparison results:")
print(f"Spearman correlation coefficient: {correlation:.4f} (p-value: {p_value:.4f})")
print(f"MDOD AUC: {mdod_auc:.4f}, LOF AUC: {lof_auc:.4f}")
print(f"Algorithm Complexity Comparison (Empirical Running Time)：MDOD {mdod_time:.6f}s vs LOF {lof_time:.6f}s")

# Test data comparison
data_df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
data_df['True_Label'] = y_true
data_df['MDOD_Score'] = mdod_scores
data_df['MDOD_Label'] = mdod_labels
data_df['LOF_Score'] = lof_scores
data_df['LOF_Label'] = lof_labels
print("\nTest Data Comparison (Top 10 Rows):")
print(data_df.head(10))
data_df.to_csv('test_data_comparison.csv', index=False)
print("The complete data has been saved to 'test_data_comparison.csv'")

# Save test output data to another file (including all metrics)
output_data = {
    'MDOD_AUC': [mdod_auc],
    'MDOD_Time': [mdod_time],
    'MDOD_Precision': [mdod_precision],
    'MDOD_Recall': [mdod_recall],
    'MDOD_F1': [mdod_f1],
    'LOF_AUC': [lof_auc],
    'LOF_Time': [lof_time],
    'LOF_Precision': [lof_precision],
    'LOF_Recall': [lof_recall],
    'LOF_F1': [lof_f1],
    'Spearman_Correlation': [correlation],
    'Spearman_PValue': [p_value]
}
output_df = pd.DataFrame(output_data)
output_df.to_csv('test_output_data.csv', index=False)
print("Test output data has been saved to 'test_output_data.csv'")

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

visualize_data(X, mdod_labels, 'MDOD Predictions', y_true)
visualize_data(X, lof_labels, 'LOF Predictions', y_true)

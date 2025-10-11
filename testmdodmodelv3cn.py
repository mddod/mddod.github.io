import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from pyod.models.lof import LOF
from mdod import MDOD  # 使用优化后的模块

# 定义测试参数（可自定义）
n_samples = 1000  # 总测试点数量
outliers_num = 150  # 异常值数量
n_features = 2  # 维度
contamination = outliers_num / n_samples  # 异常比例
norm_distance = 1.0  # MDOD参数，与radius_limit协调
top_n = 20  # MDOD参数，调整为论文建议的较小值
sampling_rate = 0.05  # 新参数，0 < sampling_rate <= 1，1 为全量
random_state = 42  # 为采样添加随机种子，可复现
LOF_n_neighbors = 100

# 生成合成数据：正常点在中心簇，限定范围，异常点散布在外围
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

# 正常数据：限定范围内的中心簇
center_sigma = 0.3
try:
    normal = generate_clustered_data(n_samples - outliers_num, n_features, 
                                   mean=np.zeros(n_features), 
                                   cov=np.eye(n_features) * center_sigma, 
                                   radius_limit=1.0)
except ValueError as e:
    print(f"Error in normal data generation: {e}")
    exit(1)

# 异常数据：散布在外围
r_min, r_max = 1.5, 3.0
outliers = np.random.randn(outliers_num, n_features)
norms = np.linalg.norm(outliers, axis=1, keepdims=True)
outliers = outliers / norms * np.random.uniform(r_min, r_max, size=(outliers_num, 1))

# 合并数据
X = np.vstack([normal, outliers])
y_true = np.concatenate([np.zeros(n_samples - outliers_num), np.ones(outliers_num)])

print(f"生成数据: {n_samples} 样本, {n_features} 维度, {outliers_num} 异常值")

# 保存生成的数据到本地文件
data_gen_df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
data_gen_df['True_Label'] = y_true
data_gen_df.to_csv('generated_data.csv', index=False)
print("生成数据已保存到 'generated_data.csv'")

# 从文件读取数据
data_gen_df = pd.read_csv('generated_data.csv')
X = data_gen_df[[f"Feature_{i+1}" for i in range(n_features)]].values
y_true = data_gen_df['True_Label'].values

# 使用MDOD并测量时间
mdod = MDOD(norm_distance=norm_distance, top_n=top_n, contamination=contamination, 
            sampling_rate=sampling_rate, random_state=random_state)
start_time = time.time()
mdod.fit(X)
mdod_time = time.time() - start_time
mdod_scores = mdod.decision_scores_
mdod_labels = mdod.labels_
mdod_auc = roc_auc_score(y_true, mdod_scores)

print(f"使用采样率: {sampling_rate}")

print("\nMDOD 决策分数 (前10个):")
print(mdod_scores[:10])
print("MDOD 预测标签 (异常数):", np.sum(mdod_labels))
print("MDOD AUC:", mdod_auc)
print(f"MDOD 运行时间: {mdod_time:.6f} 秒")

# 计算更多指标 for MDOD
mdod_cm = confusion_matrix(y_true, mdod_labels)
mdod_precision = precision_score(y_true, mdod_labels)
mdod_recall = recall_score(y_true, mdod_labels)
mdod_f1 = f1_score(y_true, mdod_labels)

print("MDOD Confusion Matrix:")
print(mdod_cm)
print(f"MDOD Precision: {mdod_precision:.4f}")
print(f"MDOD Recall: {mdod_recall:.4f}")
print(f"MDOD F1-Score: {mdod_f1:.4f}")

# 使用PYOD的LOF作为对比
lof = LOF(n_neighbors=LOF_n_neighbors, contamination=contamination)
#lof = LOF(n_neighbors=top_n, contamination=contamination)
start_time = time.time()
lof.fit(X)
lof_time = time.time() - start_time
lof_scores = lof.decision_scores_
lof_labels = lof.labels_
lof_auc = roc_auc_score(y_true, lof_scores)

print("\nLOF 决策分数 (前10个):")
print(lof_scores[:10])
print("LOF 预测标签 (异常数):", np.sum(lof_labels))
print("LOF AUC:", lof_auc)
print(f"LOF 运行时间: {lof_time:.6f} 秒")

# 计算更多指标 for LOF
lof_cm = confusion_matrix(y_true, lof_labels)
lof_precision = precision_score(y_true, lof_labels)
lof_recall = recall_score(y_true, lof_labels)
lof_f1 = f1_score(y_true, lof_labels)

print("LOF Confusion Matrix:")
print(lof_cm)
print(f"LOF Precision: {lof_precision:.4f}")
print(f"LOF Recall: {lof_recall:.4f}")
print(f"LOF F1-Score: {lof_f1:.4f}")

# 比较
correlation, p_value = spearmanr(mdod_scores, lof_scores)
print("\n比较结果:")
print(f"Spearman 相关系数: {correlation:.4f} (p-value: {p_value:.4f})")
print(f"MDOD AUC: {mdod_auc:.4f}, LOF AUC: {lof_auc:.4f}")
print(f"算法复杂度比较（实证运行时间）：MDOD {mdod_time:.6f}s vs LOF {lof_time:.6f}s")

# 测试数据对比
data_df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
data_df['True_Label'] = y_true
data_df['MDOD_Score'] = mdod_scores
data_df['MDOD_Label'] = mdod_labels
data_df['LOF_Score'] = lof_scores
data_df['LOF_Label'] = lof_labels
print("\n测试数据对比 (前10行):")
print(data_df.head(10))
data_df.to_csv('test_data_comparison.csv', index=False)
print("完整数据已保存到 'test_data_comparison.csv'")

# 保存测试输出数据到另一个文件（包含所有指标）
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
print("测试输出数据已保存到 'test_output_data.csv'")

# 可视化函数
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

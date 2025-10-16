import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from sklearn.decomposition import PCA
from mdod import MDOD  # MDOD模块

# 定义测试参数
contamination = 0.15  # 异常比例
norm_distance = 1.0  # MDOD参数
top_n = 20  # MDOD参数
sampling_rate = 0.05  # 采样率
random_state = 42  # 随机种子

# 从文件读取数据
data_gen_df = pd.read_csv('testmdod_simple_example_input_data.csv')
# 获取特征列（Feature_1, Feature_2, ...）
feature_columns = [col for col in data_gen_df.columns if col.startswith('Feature_')]
X = data_gen_df[feature_columns].values
# 获取数据数量和维数
n_samples = X.shape[0]
n_features = X.shape[1]
outliers_num = int(n_samples * contamination)  # 计算异常值数量
print(f"数据维数: {n_features}")
print(f"数据点数量: {n_samples}")
print(f"异常值数量: {outliers_num}")

# 使用MDOD并测量时间
mdod = MDOD(norm_distance=norm_distance, top_n=top_n, contamination=contamination, 
            sampling_rate=sampling_rate, random_state=random_state)
start_time = time.time()
mdod.fit(X)
mdod_time = time.time() - start_time
mdod_scores = mdod.decision_scores_
mdod_labels = mdod.labels_  # 获取MDOD预测标签

# 根据决策分数生成异常标签（前contamination比例的点为异常）
anomaly_indices = np.argsort(mdod_scores)[-outliers_num:]  # 取最大的outliers_num个索引
anomaly_labels = np.zeros(n_samples)  # 初始化全为正常（0）
anomaly_labels[anomaly_indices] = 1  # 标记异常点为1

# 将决策分数和异常标签添加到原始DataFrame
data_gen_df['MDOD_Score'] = mdod_scores  # 新增决策分数列
data_gen_df['Anomaly_Label'] = anomaly_labels  # 新增异常标签列

# 保存修改后的DataFrame到输出文件
output_file = 'testmdod_simple_example_output_data.csv'
data_gen_df.to_csv(output_file, index=False)
print(f"已将数据保存到: {output_file}")

# 打印运行时间和相关信息
print(f"使用采样率: {sampling_rate}")
print(f"MDOD 运行时间: {mdod_time:.4f} 秒")
print("\nMDOD 决策分数 (前10个 - 最大值):")
print(np.sort(mdod_scores)[-10:][::-1])  # 排序后取最大10个并倒序显示
print("\nMDOD 决策分数 (后10个 - 最小值):")
print(np.sort(mdod_scores)[:10])  # 排序后取最小10个

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

# 调用可视化函数
visualize_data(X, mdod_labels, 'MDOD Predictions', anomaly_labels)

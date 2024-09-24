import numpy as np
import matplotlib.pyplot as plt
import mdod

# 生成200个簇状分布的正常数据
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

# 生成异常数据
num_outliers = 10  # 异常数据数量为10
outliers = np.random.uniform(low=-50, high=50, size=(num_outliers, 2))

# 合并正常数据和异常数据
data_with_outliers = np.vstack([data, outliers])

# 使用mdod库对数据进行异常值检测
nd = 1  # 测试观察点在新增维度的数值
sn = 15 # 按从大到小评分值排序对前多少个进行统计
result = mdod.md(data_with_outliers, nd, sn)

# 提取异常值评分（result中的第一列）
scores = np.array([float(r[0]) for r in result])  # 提取第1列作为异常值评分

# 找到评分最低的10个数据点作为异常值
outliers_indices = np.argsort(scores)[:num_outliers]  # 找到分数最低的10个索引
normal_indices = np.setdiff1d(np.arange(len(scores)), outliers_indices)  # 找到正常值的索引

# 打印并可视化数据集
plt.figure(figsize=(10, 6))

# 绘制正常数据点
plt.scatter(data_with_outliers[normal_indices, 0], data_with_outliers[normal_indices, 1], 
            c='blue', label='Normal Data')

# 绘制异常数据点
plt.scatter(data_with_outliers[outliers_indices, 0], data_with_outliers[outliers_indices, 1], 
            c='red', label='Outliers')

# 图例与显示
plt.title('Outliers vs Normal Data (2D)')
plt.legend()
plt.show()

# import json
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from collections import Counter

# # 读取JSON数据
# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_train_data_shuffled_without_1600worse_data.json', 'r') as file:
#     data = json.load(file)

# # 提取特征向量
# texts = [item['instruction'] + ' ' + item['input'] + ' ' + item['output'] for item in data]
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(texts)

# # 聚类分析
# n_clusters = 125  # 假设选择125个聚类
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# clusters = kmeans.fit_predict(X)

# # 从每个聚类中选取数据
# cluster_counter = Counter(clusters)
# samples_per_cluster = 46476 // n_clusters
# selected_indices = []

# for cluster in range(n_clusters):
#     cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
#     if len(cluster_indices) > samples_per_cluster:
#         selected_indices.extend(np.random.choice(cluster_indices, samples_per_cluster, replace=False))
#     else:
#         selected_indices.extend(cluster_indices)

# # 选取46476条数据
# selected_data = [data[i] for i in selected_indices[:46476]]

# # 保存筛选后的数据
# with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_5%_data/all_initial_data_5%.json', 'w') as file:
#     json.dump(selected_data, file, ensure_ascii=False, indent=4)

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# 读取JSON数据
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/all_train_data_shuffled_without_1600worse_data.json', 'r') as file:
    data = json.load(file)

# 提取特征向量
texts = [item['instruction'] + ' ' + item['input'] + ' ' + item['output'] for item in data]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# 聚类分析
n_clusters = 464  # 假设选择295个聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# 从每个聚类中选取数据
cluster_counter = Counter(clusters)
samples_per_cluster = 46476 // n_clusters
selected_indices = []

for cluster in range(n_clusters):
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
    if len(cluster_indices) > samples_per_cluster:
        selected_indices.extend(np.random.choice(cluster_indices, samples_per_cluster, replace=False))
    else:
        selected_indices.extend(cluster_indices)

# 如果选取的数据少于46476条，随机补充数据
if len(selected_indices) < 46476:
    remaining_indices = [i for i in range(len(data)) if i not in selected_indices]
    additional_samples = 46476 - len(selected_indices)
    selected_indices.extend(np.random.choice(remaining_indices, additional_samples, replace=False))

# 选取46476条数据
selected_data = [data[i] for i in selected_indices[:46476]]

# 保存筛选后的数据
with open('/gluster_pdc_llm/user/songjielin/extracted_120k_data/selected_15%_data/all_initial_data_15%.json', 'w') as file:
    json.dump(selected_data, file, ensure_ascii=False, indent=4)


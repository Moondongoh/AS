# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt

# # 아이리스 데이터셋 로드
# iris = load_iris()
# X = iris.data

# # 데이터 분할
# data_1 = X[0:50]
# data_2 = X[50:100]
# data_3 = X[100:150]

# # 결과 출력
# #print(data_1)
# #print(data_2)
# #print(data_3)

# # 각 데이터셋의 평균 구하기
# mean_1 = np.mean(data_1, axis=0)
# mean_2 = np.mean(data_2, axis=0)
# mean_3 = np.mean(data_3, axis=0)

# # 결과 출력
# #print("data_1 평균:", mean_1)
# #print("data_2 평균:", mean_2)
# #print("data_3 평균:", mean_3)

# # 150개 데이터 중 3개를 임의로 선택
# random_indices = np.random.choice(len(X), 3, replace=False)
# initial_centroids = X[random_indices]

# # 선택된 초기 중심점 출력
# print("150개 중 선택된 3개의 값:")
# print(initial_centroids)

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 아이리스 데이터셋 로드
iris = load_iris()
X = iris.data

# 데이터 분할
data_1 = X[0:50]
data_2 = X[50:100]
data_3 = X[100:150]

# 각 데이터셋의 평균 구하기
mean_1 = np.mean(data_1, axis=0)
mean_2 = np.mean(data_2, axis=0)
mean_3 = np.mean(data_3, axis=0)

# 150개 데이터 중 3개를 임의로 선택
np.random.seed(42)  # 재현성을 위해 시드 고정
random_indices = np.random.choice(len(X), 3, replace=False)
initial_centroids = X[random_indices]

# 선택된 초기 중심점 출력
print("150개 중 선택된 3개의 값:")
print(initial_centroids)

def compute_distance(point, centroid):
    return np.linalg.norm(point - centroid)

def assign_clusters(X, centroids):
    clusters = {}
    for i in range(len(centroids)):
        clusters[i] = []

    for point in X:
        distances = [compute_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters[cluster].append(point)

    return clusters

def update_centroids(clusters):
    new_centroids = []
    for key in clusters.keys():
        new_centroid = np.mean(clusters[key], axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# 클러스터 할당 및 중심점 업데이트 반복
max_iterations = 150
centroids = initial_centroids

for i in range(max_iterations):
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(clusters)

    # 중심점 변화가 없으면 중지
    if np.allclose(centroids, new_centroids, atol=1e-6):
        break
    centroids = new_centroids

# 최종 중심점 출력
print("최종 중심점:")
print(centroids)

# # 각 데이터 포인트의 클러스터 할당 결과 출력
for cluster_id, points in clusters.items():
    print(f"\n클러스터 {cluster_id}:")
    print(points)

# 시각화 (선택 사항)
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for cluster_id, points in clusters.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c=colors[cluster_id], label=f'Cluster {cluster_id}')
    plt.scatter(centroids[cluster_id, 0], centroids[cluster_id, 1], c='k', marker='x')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title("Iris Data Clustering")
plt.show()

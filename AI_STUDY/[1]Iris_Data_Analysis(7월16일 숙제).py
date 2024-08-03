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

# 3개의 초기 클러스터 중심을 각 그룹의 평균으로 설정
initial_centroids = np.array([data_1.mean(axis=0), data_2.mean(axis=0), data_3.mean(axis=0)])
centroids = initial_centroids.copy()

# 유클리드 거리 계산 함수
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# 군집 할당 함수
def assign_clusters(X, centroids):
    clusters = []
    for sample in X:
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# 새로운 중심 계산 함수
def compute_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        new_centroid = cluster_points.mean(axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# K-평균 클러스터링 실행
def k_means(X, k, initial_centroids, max_iters=150):
    centroids = initial_centroids
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = compute_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# 실행 및 결과 시각화
k = 3
clusters, final_centroids = k_means(X, k, initial_centroids)

# 클러스터 결과 시각화 (첫 두 개의 특성만 사용)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering of Iris Dataset')
plt.show()

# 결과 출력
print("최종 클러스터 중심:")
print(final_centroids)

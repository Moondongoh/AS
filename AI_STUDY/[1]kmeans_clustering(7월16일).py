import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 아이리스 데이터셋 로드
iris = load_iris()
data = iris.data

# 초기 클러스터 중심 임의 설정 (데이터 범위 내에서 랜덤값 4개 속성)
np.random.seed(42)
initial_centroids = data[np.random.choice(data.shape[0], 3, replace=False)]

# 거리 계산 함수 (유클리드 거리)
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-means 클러스터링 수행
def kmeans_clustering(data, centroids, iterations=150):
    for _ in range(iterations):
        clusters = {i: [] for i in range(len(centroids))}
        
        # 각 데이터 포인트를 가장 가까운 클러스터에 할당
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)
        
        # 새로운 클러스터 중심 계산
        new_centroids = []
        for key in clusters:
            if clusters[key]:  # 빈 클러스터를 방지하기 위함
                new_centroids.append(np.mean(clusters[key], axis=0))
            else:
                new_centroids.append(centroids[key])  # 클러스터가 비어 있으면 기존 중심 유지
        
        centroids = np.array(new_centroids)
    
    return centroids, clusters

# K-means 클러스터링 수행
final_centroids, final_clusters = kmeans_clustering(data, initial_centroids)

# 결과 출력
for i, centroid in enumerate(final_centroids):
    print(f"Cluster {i+1} centroid: {centroid}")

# 클러스터링 결과 데이터프레임 생성
clustered_data = []
for cluster_id, points in final_clusters.items():
    for point in points:
        clustered_data.append(list(point) + [cluster_id])

df_clustered = pd.DataFrame(clustered_data, columns=iris.feature_names + ['cluster'])

# 데이터프레임 출력
df_clustered.head(10)  # 앞부분 10개만 출력

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# 아이리스 데이터셋 로드
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# 데이터프레임으로 변환
df = pd.DataFrame(data, columns=feature_names)
df['class'] = target

# 클래스 레이블을 a, b, c로 변환
class_mapping = {0: 'a', 1: 'b', 2: 'c'}
df['class'] = df['class'].map(class_mapping)

# 각 클래스별로 데이터 분리
df_a = df[df['class'] == 'a']
df_b = df[df['class'] == 'b']
df_c = df[df['class'] == 'c']

# 각 클래스별 평균, 분산, 표준편차 계산
mean_a = df_a[feature_names].mean()
variance_a = df_a[feature_names].var()
std_deviation_a = df_a[feature_names].std()

mean_b = df_b[feature_names].mean()
variance_b = df_b[feature_names].var()
std_deviation_b = df_b[feature_names].std()

mean_c = df_c[feature_names].mean()
variance_c = df_c[feature_names].var()
std_deviation_c = df_c[feature_names].std()

# 결과 출력
print("아이리스 데이터셋의 클래스 a 데이터:")
print(df_a)
print("\n클래스 a의 각 특성별 평균:")
print(mean_a)
print("\n클래스 a의 각 특성별 분산:")
print(variance_a)
print("\n클래스 a의 각 특성별 표준편차:")
print(std_deviation_a)

print("\n아이리스 데이터셋의 클래스 b 데이터:")
print(df_b)
print("\n클래스 b의 각 특성별 평균:")
print(mean_b)
print("\n클래스 b의 각 특성별 분산:")
print(variance_b)
print("\n클래스 b의 각 특성별 표준편차:")
print(std_deviation_b)

print("\n아이리스 데이터셋의 클래스 c 데이터:")
print(df_c)
print("\n클래스 c의 각 특성별 평균:")
print(mean_c)
print("\n클래스 c의 각 특성별 분산:")
print(variance_c)
print("\n클래스 c의 각 특성별 표준편차:")
print(std_deviation_c)
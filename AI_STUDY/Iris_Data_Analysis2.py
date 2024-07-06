import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#df['target'] = iris.target

df1 = df.iloc[:50]
df2 = df.iloc[50:100]
df3 = df.iloc[100:150]

# 각 속성의 합 계산
sum_df1 = df1.sum()
sum_df2 = df2.sum()
sum_df3 = df3.sum()

# 각 속성의 평균 계산
mean_df1 = sum_df1 / 50
mean_df2 = sum_df2 / 50
mean_df3 = sum_df3 / 50

print("********************평균********************")
print("Data A의 50개 평균:")
print(mean_df1)

print("\nData B의 50개 평균:")
print(mean_df2)

print("\nData C의 50개 평균:")
print(mean_df3)
print("\n")

# 각 값에서 평균을 빼고 제곱 후 모두 더한 뒤 전체 데이터 수로 나눠줌.
def calculate_variance(data):
    mean = data.mean()
    variance = np.sum((data - mean) ** 2) / len(data)
    return variance

# 각 속성의 분산 계산
var_df1 = df1.apply(calculate_variance)
var_df2 = df2.apply(calculate_variance)
var_df3 = df3.apply(calculate_variance)

print("********************분산********************")
print("\nData A의 50개 분산 값:")
print(var_df1)

print("\nData A의 50개 분산 값:")
print(var_df2)

print("\nData A의 50개 분산 값:")
print(var_df3)
print("\n")

# 표준편차 루트 씌우기 sqrt
def calculate_variance_root(variance):
    return np.sqrt(variance)

# 각 속성의 분산 계산
var_df4 = var_df1.apply(calculate_variance_root)
var_df5 = var_df2.apply(calculate_variance_root)
var_df6 = var_df3.apply(calculate_variance_root)

print("********************표준편차********************")
print("\nData A의 50개 표준편차 값:")
print(var_df4)

print("\nData A의 50개 표준편차 값:")
print(var_df5)

print("\nData A의 50개 표준편차 값:")
print(var_df6)
# IRIS DATA 값을 활용해 각 파트별 평균, 분산, 표준편차를 구하는 코드
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import euclidean

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

#데이터 내용
print("************************************************************")
print(df)
print("************************************************************")

#30개
df1 = df.iloc[:30]
df2 = df.iloc[50:80]
df3 = df.iloc[100:130]

#20개
df1_1 = df.iloc[30:50]
df2_2 = df.iloc[80:100]
df3_3 = df.iloc[130:150]

# 30개
sum_df1 = df1.sum()
sum_df2 = df2.sum()
sum_df3 = df3.sum()

# 20개
sum_df1_1 = df1.sum()
sum_df2_2 = df2.sum()
sum_df3_3 = df3.sum()

# 30개
mean_df1 = sum_df1 / 50
mean_df2 = sum_df2 / 50
mean_df3 = sum_df3 / 50

# 20개
mean_df1_1 = sum_df1_1 / 50
mean_df2_2 = sum_df2_2 / 50
mean_df3_3 = sum_df3_3 / 50

print("********************평균********************")
print("Data A의 30개 평균:")
print(mean_df1)

print("\nData B의 30개 평균:")
print(mean_df2)

print("\nData C의 30개 평균:")
print(mean_df3)
print("\n")

# 각 값에서 평균을 빼고 제곱 후 모두 더한 뒤 전체 데이터 수로 나눠줌.
def calculate_variance(data):
    mean = data.mean()
    #모든거 더하기 np.sum
    variance = np.sum((data - mean) ** 2) / len(data)
    return variance

# 각 속성의 분산 계산
var_df1 = df1.apply(calculate_variance)
var_df2 = df2.apply(calculate_variance)
var_df3 = df3.apply(calculate_variance)

print("********************분산********************")
print("\nData A의 30개 분산 값:")
print(var_df1)

print("\nData B의 30개 분산 값:")
print(var_df2)

print("\nData B의 30개 분산 값:")
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
print("\nData A의 30개 표준편차 값:")
print(var_df4)

print("\nData B의 30개 표준편차 값:")
print(var_df5)

print("\nData C의 30개 표준편차 값:")
print(var_df6)

print("-----------------------------------------------\n")
# 각 값에서 평균을 빼고 제곱 후 모두 더한 뒤 전체 데이터 수로 나눠줌.
def calculate_variance(data):
    mean = data.mean()
    #모든거 더하기 np.sum
    variance = np.sum((data - mean) ** 2) / len(data)
    return variance

# 각 속성의 분산 계산
var_df1_1 = df1_1.apply(calculate_variance)
var_df2_2 = df2_2.apply(calculate_variance)
var_df3_3 = df3_3.apply(calculate_variance)

# 표준편차 루트 씌우기 sqrt
def calculate_variance_root(variance):
    return np.sqrt(variance)

# 각 속성의 분산 계산
var_df4_4 = var_df1_1.apply(calculate_variance_root)
var_df5_5 = var_df2_2.apply(calculate_variance_root)
var_df6_6 = var_df3_3.apply(calculate_variance_root)

print("********************평균********************")
print("Data A의 20개 평균:")
print(mean_df1_1)

print("\nData A의 20개 분산 값:")
print(var_df1_1)

print("\nData A의 20개 표준편차 값:")
print(var_df4_4)

print("\n")

print("\nData B의 20개 평균:")
print(mean_df2_2)

print("\nData B의 20개 분산 값:")
print(var_df2_2)

print("\nData B의 20개 표준편차 값:")
print(var_df5_5)

print("\n")

print("\nData C의 20개 평균:")
print(mean_df3_3)

print("\nData C의 20개 분산 값:")
print(var_df3_3)

print("\nData C의 20개 표준편차 값:")
print(var_df6_6)
print("\n")

print("-----------------------------------------------\n")

# 각 그룹의 평균 계산 (sepal length 속성만 사용)
mean_df1_sepal_length = df1['sepal length (cm)'].mean()
mean_df2_sepal_length = df2['sepal length (cm)'].mean()
mean_df3_sepal_length = df3['sepal length (cm)'].mean()

mean_df1_1_sepal_length = df1_1['sepal length (cm)'].mean()
mean_df2_2_sepal_length = df2_2['sepal length (cm)'].mean()
mean_df3_3_sepal_length = df3_3['sepal length (cm)'].mean()

# 유클리드 거리 계산 함수
def calculate_euclidean_distance(mean1, mean2):
    return euclidean([mean1], [mean2])

# 가장 가까운 거리 찾기 함수
def find_closest(mean, comparisons):
    min_distance = float('inf')
    closest_group = None
    for group, mean_value in comparisons.items():
        distance = calculate_euclidean_distance(mean, mean_value)
        if distance < min_distance:
            min_distance = distance
            closest_group = group
    return closest_group, min_distance

# 데이터 20개 각각을 30개 데이터 그룹과 비교
def compare_to_groups(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        closest_group, min_distance = find_closest(row['sepal length (cm)'], means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 sepal length 비교
mean_30 = {
    "A (30)": mean_df1_sepal_length,
    "B (30)": mean_df2_sepal_length,
    "C (30)": mean_df3_sepal_length,
}

# 20개 데이터 각각 비교
results_A = compare_to_groups(df1_1, mean_30)
results_B = compare_to_groups(df2_2, mean_30)
results_C = compare_to_groups(df3_3, mean_30)

# 결과 출력
print("******************** sepal_length 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    
print("-----------------------------------------------\n")

# 각 그룹의 평균 계산 (sepal width 속성만 사용)
mean_df1_sepal_width = df1['sepal width (cm)'].mean()
mean_df2_sepal_width = df2['sepal width (cm)'].mean()
mean_df3_sepal_width = df3['sepal width (cm)'].mean()

mean_df1_1_sepal_width = df1_1['sepal width (cm)'].mean()
mean_df2_2_sepal_width = df2_2['sepal width (cm)'].mean()
mean_df3_3_sepal_width = df3_3['sepal width (cm)'].mean()

# 데이터 20개 각각을 30개 데이터 그룹과 비교
def compare_to_groups_sepal_width(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        closest_group, min_distance = find_closest(row['sepal width (cm)'], means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 sepal width 비교
mean_30_width = {
    "A (30)": mean_df1_sepal_width,
    "B (30)": mean_df2_sepal_width,
    "C (30)": mean_df3_sepal_width,
}

# 20개 데이터 각각 비교
results_A_width = compare_to_groups_sepal_width(df1_1, mean_30_width)
results_B_width = compare_to_groups_sepal_width(df2_2, mean_30_width)
results_C_width = compare_to_groups_sepal_width(df3_3, mean_30_width)

# 결과 출력
print("******************** sepal_width 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A_width:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B_width:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C_width:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    
print("-----------------------------------------------\n")

# 각 그룹의 평균 계산 (petal length 속성만 사용)
mean_df1_petal_length = df1['petal length (cm)'].mean()
mean_df2_petal_length = df2['petal length (cm)'].mean()
mean_df3_petal_length = df3['petal length (cm)'].mean()

mean_df1_1_petal_length = df1_1['petal length (cm)'].mean()
mean_df2_2_petal_length = df2_2['petal length (cm)'].mean()
mean_df3_3_petal_length = df3_3['petal length (cm)'].mean()

# 데이터 20개 각각을 30개 데이터 그룹과 비교
def compare_to_groups_petal_length(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        closest_group, min_distance = find_closest(row['petal length (cm)'], means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 petal length 비교
mean_30_petal_length = {
    "A (30)": mean_df1_petal_length,
    "B (30)": mean_df2_petal_length,
    "C (30)": mean_df3_petal_length,
}

# 20개 데이터 각각 비교
results_A_petal_length = compare_to_groups_petal_length(df1_1, mean_30_petal_length)
results_B_petal_length = compare_to_groups_petal_length(df2_2, mean_30_petal_length)
results_C_petal_length = compare_to_groups_petal_length(df3_3, mean_30_petal_length)

# 결과 출력
print("******************** petal_length 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A_petal_length:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B_petal_length:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C_petal_length:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    
print("-----------------------------------------------\n")

# 각 그룹의 평균 계산 (petal width 속성만 사용)
mean_df1_petal_width = df1['petal width (cm)'].mean()
mean_df2_petal_width = df2['petal width (cm)'].mean()
mean_df3_petal_width = df3['petal width (cm)'].mean()

mean_df1_1_petal_width = df1_1['petal width (cm)'].mean()
mean_df2_2_petal_width = df2_2['petal width (cm)'].mean()
mean_df3_3_petal_width = df3_3['petal width (cm)'].mean()

# 데이터 20개 각각을 30개 데이터 그룹과 비교
def compare_to_groups_petal_width(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        closest_group, min_distance = find_closest(row['petal width (cm)'], means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 petal width 비교
mean_30_petal_width = {
    "A (30)": mean_df1_petal_width,
    "B (30)": mean_df2_petal_width,
    "C (30)": mean_df3_petal_width,
}

# 20개 데이터 각각 비교
results_A_petal_width = compare_to_groups_petal_width(df1_1, mean_30_petal_width)
results_B_petal_width = compare_to_groups_petal_width(df2_2, mean_30_petal_width)
results_C_petal_width = compare_to_groups_petal_width(df3_3, mean_30_petal_width)

# 결과 출력
print("******************** petal_width 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A_petal_width:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B_petal_width:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C_petal_width:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    
print("[2개]-----------------------------------------------\n")

# 각 그룹의 평균 계산 (sepal length와 sepal width 속성 모두 사용)
mean_df1_sepal_length = df1['sepal length (cm)'].mean()
mean_df1_sepal_width = df1['sepal width (cm)'].mean()
mean_df2_sepal_length = df2['sepal length (cm)'].mean()
mean_df2_sepal_width = df2['sepal width (cm)'].mean()
mean_df3_sepal_length = df3['sepal length (cm)'].mean()
mean_df3_sepal_width = df3['sepal width (cm)'].mean()

mean_df1_1_sepal_length = df1_1['sepal length (cm)'].mean()
mean_df1_1_sepal_width = df1_1['sepal width (cm)'].mean()
mean_df2_2_sepal_length = df2_2['sepal length (cm)'].mean()
mean_df2_2_sepal_width = df2_2['sepal width (cm)'].mean()
mean_df3_3_sepal_length = df3_3['sepal length (cm)'].mean()
mean_df3_3_sepal_width = df3_3['sepal width (cm)'].mean()

# 유클리드 거리 계산 함수 (여러 속성 고려)
def calculate_euclidean_distance(point1, point2):
    return euclidean(point1, point2)

# 가장 가까운 거리 찾기 함수 (여러 속성 고려)
def find_closest(point, comparisons):
    min_distance = float('inf')
    closest_group = None
    for group, mean_point in comparisons.items():
        distance = calculate_euclidean_distance(point, mean_point)
        if distance < min_distance:
            min_distance = distance
            closest_group = group
    return closest_group, min_distance

# 데이터 20개 각각을 30개 데이터 그룹과 비교 (sepal length와 sepal width를 함께 사용)
def compare_to_groups(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        point = (row['sepal length (cm)'], row['sepal width (cm)'])
        closest_group, min_distance = find_closest(point, means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 sepal length와 sepal width 비교
mean_30 = {
    "A (30)": (mean_df1_sepal_length, mean_df1_sepal_width),
    "B (30)": (mean_df2_sepal_length, mean_df2_sepal_width),
    "C (30)": (mean_df3_sepal_length, mean_df3_sepal_width),
}

# 20개 데이터 각각 비교
results_A = compare_to_groups(df1_1, mean_30)
results_B = compare_to_groups(df2_2, mean_30)
results_C = compare_to_groups(df3_3, mean_30)

# 결과 출력
print("******************** sepal_length와 sepal_width 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    

print("[2개]-----------------------------------------------\n")

# 각 그룹의 평균 계산 (sepal length와 petal width 속성 모두 사용)
mean_df1_sepal_length = df1['sepal length (cm)'].mean()
mean_df1_petal_width = df1['petal width (cm)'].mean()
mean_df2_sepal_length = df2['sepal length (cm)'].mean()
mean_df2_petal_width = df2['petal width (cm)'].mean()
mean_df3_sepal_length = df3['sepal length (cm)'].mean()
mean_df3_petal_width = df3['petal width (cm)'].mean()

mean_df1_1_sepal_length = df1_1['sepal length (cm)'].mean()
mean_df1_1_petal_width = df1_1['petal width (cm)'].mean()
mean_df2_2_sepal_length = df2_2['sepal length (cm)'].mean()
mean_df2_2_petal_width = df2_2['petal width (cm)'].mean()
mean_df3_3_sepal_length = df3_3['sepal length (cm)'].mean()
mean_df3_3_petal_width = df3_3['petal width (cm)'].mean()

# 유클리드 거리 계산 함수 (여러 속성 고려)
def calculate_euclidean_distance(point1, point2):
    return euclidean(point1, point2)

# 가장 가까운 거리 찾기 함수 (여러 속성 고려)
def find_closest(point, comparisons):
    min_distance = float('inf')
    closest_group = None
    for group, mean_point in comparisons.items():
        distance = calculate_euclidean_distance(point, mean_point)
        if distance < min_distance:
            min_distance = distance
            closest_group = group
    return closest_group, min_distance

# 데이터 20개 각각을 30개 데이터 그룹과 비교 (sepal length와 sepal width를 함께 사용)
def compare_to_groups(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        point = (row['sepal length (cm)'], row['petal width (cm)'])
        closest_group, min_distance = find_closest(point, means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 sepal length와 petal width 비교
mean_30 = {
    "A (30)": (mean_df1_sepal_length, mean_df1_petal_width),
    "B (30)": (mean_df2_sepal_length, mean_df2_petal_width),
    "C (30)": (mean_df3_sepal_length, mean_df3_petal_width),
}

# 20개 데이터 각각 비교
results_A = compare_to_groups(df1_1, mean_30)
results_B = compare_to_groups(df2_2, mean_30)
results_C = compare_to_groups(df3_3, mean_30)

# 결과 출력
print("******************** sepal_length와 petal width 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")
    
print("[4개]-----------------------------------------------\n")

# 각 그룹의 평균 계산 (sepal length, sepal width, petal length, petal width 속성 모두 사용)
mean_df1 = (
    df1['sepal length (cm)'].mean(), 
    df1['sepal width (cm)'].mean(), 
    df1['petal length (cm)'].mean(), 
    df1['petal width (cm)'].mean()
)
mean_df2 = (
    df2['sepal length (cm)'].mean(), 
    df2['sepal width (cm)'].mean(), 
    df2['petal length (cm)'].mean(), 
    df2['petal width (cm)'].mean()
)
mean_df3 = (
    df3['sepal length (cm)'].mean(), 
    df3['sepal width (cm)'].mean(), 
    df3['petal length (cm)'].mean(), 
    df3['petal width (cm)'].mean()
)

mean_df1_1 = (
    df1_1['sepal length (cm)'].mean(), 
    df1_1['sepal width (cm)'].mean(), 
    df1_1['petal length (cm)'].mean(), 
    df1_1['petal width (cm)'].mean()
)
mean_df2_2 = (
    df2_2['sepal length (cm)'].mean(), 
    df2_2['sepal width (cm)'].mean(), 
    df2_2['petal length (cm)'].mean(), 
    df2_2['petal width (cm)'].mean()
)
mean_df3_3 = (
    df3_3['sepal length (cm)'].mean(), 
    df3_3['sepal width (cm)'].mean(), 
    df3_3['petal length (cm)'].mean(), 
    df3_3['petal width (cm)'].mean()
)

# 유클리드 거리 계산 함수 (여러 속성 고려)
def calculate_euclidean_distance(point1, point2):
    return euclidean(point1, point2)

# 가장 가까운 거리 찾기 함수 (여러 속성 고려)
def find_closest(point, comparisons):
    min_distance = float('inf')
    closest_group = None
    for group, mean_point in comparisons.items():
        distance = calculate_euclidean_distance(point, mean_point)
        if distance < min_distance:
            min_distance = distance
            closest_group = group
    return closest_group, min_distance

# 데이터 20개 각각을 30개 데이터 그룹과 비교 (네 가지 속성 모두 사용)
def compare_to_groups(data, means):
    closest_groups = []
    for index, row in data.iterrows():
        point = (
            row['sepal length (cm)'], 
            row['sepal width (cm)'], 
            row['petal length (cm)'], 
            row['petal width (cm)']
        )
        closest_group, min_distance = find_closest(point, means)
        closest_groups.append((index, closest_group, min_distance))
    return closest_groups

# 각 30개 평균과 20개 데이터의 sepal length, sepal width, petal length, petal width 비교
mean_30 = {
    "A (30)": mean_df1,
    "B (30)": mean_df2,
    "C (30)": mean_df3,
}

# 20개 데이터 각각 비교
results_A = compare_to_groups(df1_1, mean_30)
results_B = compare_to_groups(df2_2, mean_30)
results_C = compare_to_groups(df3_3, mean_30)

# 결과 출력
print("******************** 모든 속성(sepal_length, sepal_width, petal_length, petal_width) 20개 데이터와 30개 평균 비교 ********************")
print("Data A의 20개 데이터가 30개 평균과의 비교:")
for res in results_A:
    print(f"A {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData B의 20개 데이터가 30개 평균과의 비교:")
for res in results_B:
    print(f"B {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("\nData C의 20개 데이터가 30개 평균과의 비교:")
for res in results_C:
    print(f"C {res[0]}: {res[1]}의 30개 평균과 가장 근접하다 (거리: {res[2]:.4f})")

print("-----------------------------------------------\n")
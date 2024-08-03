import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 아이리스 데이터셋 로드
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 데이터를 각각 50개씩 3그룹으로 나눔
a = iris_df.iloc[:50]
b = iris_df.iloc[50:100]
c = iris_df.iloc[100:150]

# 각 그룹을 다시 30개와 20개로 나눔 random_state=1 뒤에 넣으면 동일 값 유지 가능
a_30, a_20 = train_test_split(a, test_size=20)
b_30, b_20 = train_test_split(b, test_size=20)
c_30, c_20 = train_test_split(c, test_size=20)

# 각 그룹에서 나눈 30개의 데이터로 회귀 분석 수행 함수
def perform_regression_A(group_30):
    X = group_30[['petal length (cm)']]
    y = group_30['petal width (cm)']
    
    reg = LinearRegression().fit(X, y)
    
    # 'coef' 회귀 계수, 'intercept' 절편
    regression_equation = f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}"
    return regression_equation

# 각 그룹의 회귀 방정식 계산
regression_a = perform_regression_A(a_30)
regression_b = perform_regression_A(b_30)
regression_c = perform_regression_A(c_30)

# 결과 출력
print("Group A 회귀 방정식:", regression_a)
print("Group B 회귀 방정식:", regression_b)
print("Group C 회귀 방정식:", regression_c)


# 각 그룹에서 나눈 30개의 데이터로 회귀 분석 수행 함수
def perform_regression_B(group_30):
    X = group_30[['petal length (cm)']]
    y = group_30['petal width (cm)']
    reg = LinearRegression().fit(X, y)
    return reg

# 각 그룹의 회귀 모델 생성
reg_a = perform_regression_B(a_30)
reg_b = perform_regression_B(b_30)
reg_c = perform_regression_B(c_30)

# 각 그룹의 20개 데이터에 대한 예측과 실제값 차이 계산 함수
def calculate_errors(group_20, reg):
    X_test = group_20[['petal length (cm)']]
    y_true = group_20['petal width (cm)']
    y_pred = reg.predict(X_test)
    errors = (y_true - y_pred) ** 2
    return errors.sum()

# 각 그룹의 오차 계산
error_a = calculate_errors(a_20, reg_a)
error_b = calculate_errors(b_20, reg_b)
error_c = calculate_errors(c_20, reg_c)

# 결과 출력
print("Group A 총 제곱 오차:", error_a)
print("Group B 총 제곱 오차:", error_b)
print("Group C 총 제곱 오차:", error_c)

# 가장 오차가 적은 그룹 찾기
min_error = min(error_a, error_b, error_c)
if min_error == error_a:
    print("Group A가 최소 오차.")
elif min_error == error_b:
    print("Group B가 최소 오차.")
else:
    print("Group C가 최소 오차 .")

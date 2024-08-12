import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로딩
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype(int)  # 레이블을 정수형으로 변환

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 데이터 표준화

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 훈련 셋과 테스트 셋의 일부 출력
print("훈련 셋의 일부 데이터와 레이블:")
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train.values  # 레이블을 추가
print(train_df.head())  # 훈련 셋의 첫 5개 샘플의 특징 데이터와 레이블

print("\n테스트 셋의 일부 데이터와 레이블:")
test_df = pd.DataFrame(X_test)
test_df['label'] = y_test.values  # 레이블을 추가
print(test_df.head())  # 테스트 셋의 첫 5개 샘플의 특징 데이터와 레이블

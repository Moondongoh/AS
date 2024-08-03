import numpy as np
from sklearn.linear_model import LinearRegression

# 주어진 데이터
x_data = np.array([50, 37, 45, 20, 31]).reshape(-1, 1)  # x 값을 2D 배열로 변환
y_data = np.array([47, 40, 50, 60, 50])  # y 값

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x_data, y_data)

# 새로운 x 값 (x=40)에 대한 y 값 예측
x_new = np.array([[40]])
y_pred = model.predict(x_new)

print(f"x=40일 때의 예측된 y 값: {y_pred[0]:.2f}")

# 회귀식의 기울기와 절편 출력
print(f"회귀식의 기울기(슬로프): {model.coef_[0]:.2f}")
print(f"회귀식의 절편: {model.intercept_:.2f}")

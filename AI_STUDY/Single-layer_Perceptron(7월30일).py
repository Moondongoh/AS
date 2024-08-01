# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# class Perceptron:
#     def __init__(self, input_size, lr=0.01, epochs=10000):
#         # 가중치 W를 0.1에서 0.9사이로 초기화 (입력값 2개와 편향을 위한 1개)
#         self.W = np.random.uniform(0.1, 0.9, input_size + 1)
#         self.lr = lr
#         # 학습 단위, 반복 횟수
#         self.epochs = epochs
#         self.history = []  # 학습 과정에서 가중치의 변화를 저장할 리스트

#     def activation_fn(self, x):
#         # 시그모이드 활성화 함수
#         return 1 / (1 + np.exp(-x))

#     def predict(self, x):
#         # 입력 데이터에 대해 예측 값 계산
#         z = self.W.T.dot(np.insert(x, 0, 1))  # 편향을 위한 1 추가
#         a = self.activation_fn(z)
#         return a

#     def fit(self, X, d):
#         # 학습 데이터에 대해 모델 학습
#         for epoch in range(self.epochs):
#             for i in range(d.shape[0]):
#                 x = np.insert(X[i], 0, 1)  # 편향 b를 위한 입력 값 추가
#                 z = self.W.T.dot(x)
#                 y = self.activation_fn(z)
#                 e = d[i] - y
#                 # self.lr는 학습률, e는 오차, x는 입력 데이터, y는 예측 값, y * (1 - y) << 시그모이드 미분 값
#                 self.W = self.W + self.lr * e * x * y * (1 - y)
#                 # 가중치의 변화를 기록
#                 self.history.append(self.W.copy())

# # x1이 0.0일 때 x2가 0.0부터 0.9까지, x1이 0.1일 때 x2가 0.0부터 0.9까지 총 100개 데이터 생성
# X = np.array([[x1, x2] for x1 in np.arange(0.0, 1.0, 0.1) for x2 in np.arange(0.0, 1.0, 0.1)])

# # 각 게이트의 정답
# d_and = np.array([int(x1 > 0.5 and x2 > 0.5) for x1, x2 in X])
# d_or = np.array([int(x1 > 0.1 or x2 > 0.1) for x1, x2 in X])

# # 퍼셉트론 모델 생성
# and_perceptron = Perceptron(input_size=2, epochs=10000)
# or_perceptron = Perceptron(input_size=2, epochs=10000)

# # 모델 학습
# print("AND 값:")
# and_perceptron.fit(X, d_and)

# # AND 게이트의 예측 값 출력 (입력 데이터와 함께)
# for idx, (x1, x2) in enumerate(X[:100]):
#     prediction = and_perceptron.predict([x1, x2])
#     print(f"#{idx + 1} Input: ({x1:.3f}, {x2:.3f}), Prediction: {prediction * 100:.1f}%")

# print("\nOR 값:")
# or_perceptron.fit(X, d_or)

# # OR 게이트의 예측 값 출력 (입력 데이터와 함께)
# for idx, (x1, x2) in enumerate(X[:100]):
#     prediction = or_perceptron.predict([x1, x2])
#     print(f"#{idx + 1} Input: ({x1:.3f}, {x2:.3f}), Prediction: {prediction * 100:.1f}%")

# # 예측 값 생성
# predictions_and = np.array([and_perceptron.predict(x) for x in X])
# predictions_or = np.array([or_perceptron.predict(x) for x in X])

# # 출력 값을 그래프로 표시
# def plot_output_values(X, predictions, title):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     X1 = X[:, 0]
#     X2 = X[:, 1]
#     Y = predictions

#     scatter = ax.scatter(X1, X2, Y, c=Y, cmap='viridis')
#     fig.colorbar(scatter, ax=ax, label='Output')
    
#     ax.set_xlabel('Input 1')
#     ax.set_ylabel('Input 2')
#     ax.set_zlabel('Output')
#     ax.set_title(title)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_zlim(0, 1)
#     ax.set_xticks(np.arange(0.0, 1.1, 0.1))
#     ax.set_yticks(np.arange(0.0, 1.1, 0.1))
#     ax.set_zticks(np.arange(0.0, 1.1, 0.1))

#     plt.show()

# # AND 게이트의 출력 값 그래프
# plot_output_values(X, predictions_and, 'AND Gate')

# # OR 게이트의 출력 값 그래프
# plot_output_values(X, predictions_or, 'OR Gate')

#***********************************************************

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    #input_size: 입력 변수의 수, lr: 학습률, epochs: 학습 반복 횟수
    # self.W: 초기 가중치 벡터, 0.1에서 0.9 사이의 난수로 초기화. 크기는 입력 변수의 수 + 1(편향 항)
    # self.history: 학습 과정에서 가중치 변화를 저장할 리스트
    def __init__(self, input_size, lr=0.01, epochs=10000):
        # 가중치 W를 0.1에서 0.9 사이로 초기화 (입력 크기 + 편향을 위한 1개)
        self.W = np.random.uniform(0.1, 0.9, input_size + 1)
        self.lr = lr
        self.epochs = epochs
        self.history = []  # 학습 과정에서 가중치 변화를 저장할 리스트

    def activation_fn(self, x):
        # 시그모이드 함수를 반환
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        # 입력 x에 대해 출력 계산
        z = self.W.T.dot(np.insert(x, 0, 1))  # 편향 항을 위한 1 추가
        return self.activation_fn(z)

    def compute_error(self, predicted, actual):
        # 실제 - 예측으로 오차 구하기
        return actual - predicted

    def update_weights(self, x, error, y):
        # 가중치 업데이트
        x = np.insert(x, 0, 1)  # 편향 항을 위한 입력에 1 추가
        gradient = self.lr * error * y * (1 - y) * x
        self.W += gradient
        self.history.append(self.W.copy())

    def predict(self, x):
        # 예측 값
        return self.feed_forward(x)

    def fit(self, X, d):
        # 주어진 데이터로 모델 학습
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.feed_forward(x)
                error = self.compute_error(y, d[i])
                self.update_weights(x, error, y)

# AND 및 OR 게이트의 학습 데이터를 생성
X = np.array([[x1, x2] for x1 in np.arange(0.0, 1.0, 0.1) for x2 in np.arange(0.0, 1.0, 0.1)])

# 둘다 0.5 이상 참 아니면 그짓 이게 목표값(정답지)
d_and = np.array([int(x1 > 0.5 and x2 > 0.5) for x1, x2 in X])

# 하나만 0.1이상 참 아니면 그짓(정답지) 
d_or = np.array([int(x1 > 0.1 or x2 > 0.1) for x1, x2 in X])

# 퍼셉트론 모델 초기화 및 학습
and_perceptron = Perceptron(input_size=2, epochs=10000)
or_perceptron = Perceptron(input_size=2, epochs=10000)

# AND 게이트 모델 학습
print("AND 게이트 학습:")
and_perceptron.fit(X, d_and)

# AND 게이트의 예측 값 및 오차 출력 (입력 데이터와 함께)
print("\nAND 게이트의 예측 값과 오차:")
for idx, (x1, x2) in enumerate(X):
    prediction = and_perceptron.predict([x1, x2])
    actual = d_and[idx]
    error = and_perceptron.compute_error(prediction, actual)
    print(f"#{idx + 1} 입력: ({x1:.3f}, {x2:.3f}), 예측 값: {prediction * 100:.1f}%, 오차: {error:.4f}")

# OR 게이트 모델 학습
print("OR 게이트 학습:")
or_perceptron.fit(X, d_or)

# OR 게이트의 예측 값 및 오차 출력 (입력 데이터와 함께)
print("\nOR 게이트의 예측 값과 오차:")
for idx, (x1, x2) in enumerate(X):
    prediction = or_perceptron.predict([x1, x2])
    actual = d_or[idx]
    error = or_perceptron.compute_error(prediction, actual)
    print(f"#{idx + 1} 입력: ({x1:.3f}, {x2:.3f}), 예측 값: {prediction * 100:.1f}%, 오차: {error:.4f}")

# 예측 값 그래프 출력 함수
def plot_output_values(X, predictions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1 = X[:, 0]
    X2 = X[:, 1]
    Y = predictions

    scatter = ax.scatter(X1, X2, Y, c=Y, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Output')

    ax.set_xlabel('input 1')
    ax.set_ylabel('input 2')
    ax.set_zlabel('output')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_zticks(np.arange(0.0, 1.1, 0.1))

    plt.show()

# AND 및 OR 게이트 예측 값 계산
predictions_and = np.array([and_perceptron.predict(x) for x in X])
predictions_or = np.array([or_perceptron.predict(x) for x in X])

print("\n************************\n")

# AND 게이트 예측 값 중 가장 높은 값 출력 (입력 값과 함께)
max_idx_and = np.argmax(predictions_and)
max_prediction_and = predictions_and[max_idx_and]
print(f"AND 게이트 예측 값 중 가장 높은 값: {max_prediction_and:.4f}, 입력 값: {X[max_idx_and]}")

# OR 게이트 예측 값 중 가장 높은 값 출력 (입력 값과 함께)
max_idx_or = np.argmax(predictions_or)
max_prediction_or = predictions_or[max_idx_or]
print(f"OR 게이트 예측 값 중 가장 높은 값: {max_prediction_or:.4f}, 입력 값: {X[max_idx_or]}")

# AND 게이트의 출력 값 그래프
plot_output_values(X, predictions_and, 'AND Gate')
# OR 게이트의 출력 값 그래프
plot_output_values(X, predictions_or, 'OR Gate')
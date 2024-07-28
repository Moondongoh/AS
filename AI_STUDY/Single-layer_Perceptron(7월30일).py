import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10000):
        # 가중치 W를 0.1에서 0.9사이로 초기화 x1과x2그리고 편향을 위한 +1
        self.W = np.random.uniform(0.1, 0.9, input_size + 1)
        self.lr = lr
        # 음 학습 단위? 반복 번수?라 생각하면됌
        self.epochs = epochs
        self.history = []  # 학습 과정에서 가중치의 변화를 저장할 리스트

    def activation_fn(self, x):
        # 시그모이드 활성화 함수
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        # 입력 데이터에 대해 예측 값 계산
        z = self.W.T.dot(np.insert(x, 0, 1))  # 편향을 위한 1 추가
        a = self.activation_fn(z)
        return a

    # def predict(self, x):
    #     # 입력 데이터에 대해 예측 값 계산
    #     z = self.W.T.dot(np.insert(x, 0, 1))  # 편향을 위한 1 추가
    #     a = self.activation_fn(z)
    #     return 1 if a >= 0.5 else 0


    def fit(self, X, d):
        # 학습 데이터에 대해 모델 학습
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)  # 편향 b를 위한 입력 값 추가
                z = self.W.T.dot(x)
                y = self.activation_fn(z)
                e = d[i] - y
                # self.lr는 학습률, e는 오차, x는 입력 데이터, y는 예측 값, y * (1 - y) << 시그모이드 미분 값
                self.W = self.W + self.lr * e * x * y * (1 - y)
                # 가중치의 변화를 기록
                self.history.append(self.W.copy())

# x1과 x2에 대한 입력 값 생성(arange를 이용해 0.1에서 1.0까지 0.1씩 늘림 x1과x2 해당)
X = np.array([[x1, x2] for x1 in np.arange(0.1, 1.0, 0.1) for x2 in np.arange(0.1, 1.0, 0.1)])

# AND 및 OR 게이트에 대한 원하는 출력 값 생성
d_and = np.array([int(x1 > 0.5 and x2 > 0.5) for x1, x2 in X])
d_or = np.array([int(x1 > 0.1 or x2 > 0.1) for x1, x2 in X])

# 퍼셉트론 모델 생성
and_perceptron = Perceptron(input_size=2, epochs=10000)

or_perceptron = Perceptron(input_size=2, epochs=10000)

# 모델 학습
print("Training AND Perceptron:")
and_perceptron.fit(X, d_and)

# # AND 퍼셉트론의 초기 10개 가중치 값 출력 (소수점 3자리)
# print("First 10 weights from AND Perceptron history (rounded to 3 decimal places):")
# for weights in and_perceptron.history[:10]:
#     rounded_weights = [f"{weight:.3f}" for weight in weights]
#     print(rounded_weights)

# AND 게이트의 첫 10개 예측 값 출력 (입력 데이터와 함께)
print("\nFirst 10 predictions from AND Gate with inputs (rounded to 3 decimal places):")
print("AND 게이트에서 Prediciction 값이 0.5이상이라면 1이다.")
for i in range(10):
    x1, x2 = X[i]
    prediction = and_perceptron.history[i][0]  # 첫 번째 요소를 추출하여 사용
    print(f"Input: ({x1:.3f}, {x2:.3f}), Prediction: {prediction:.3f}")

print("\nTraining OR Perceptron:")
or_perceptron.fit(X, d_or)

# # OR 퍼셉트론의 초기 10개 가중치 값 출력 (소수점 3자리)
# print("\nFirst 10 weights from OR Perceptron history (rounded to 3 decimal places):")
# for weights in or_perceptron.history[:10]:
#     rounded_weights = [f"{weight:.3f}" for weight in weights]
#     print(rounded_weights)

# OR 게이트의 첫 10개 예측 값 출력 (입력 데이터와 함께)
print("\nFirst 10 predictions from OR Gate with inputs (rounded to 3 decimal places):")
print("OR 게이트에서 Prediciction 값이 0.1이상이라면 1이다.")
for i in range(10):
    x1, x2 = X[i]
    prediction = or_perceptron.history[i][0]  # 첫 번째 요소를 추출하여 사용
    print(f"Input: ({x1:.3f}, {x2:.3f}), Prediction: {prediction:.3f}")

# 예측 값 생성
predictions_and = np.array([and_perceptron.predict(x) for x in X])
predictions_or = np.array([or_perceptron.predict(x) for x in X])

# 출력 값을 그래프로 표시
def plot_output_values(X, predictions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X1 = X[:, 0]
    X2 = X[:, 1]
    Y = predictions

    scatter = ax.scatter(X1, X2, Y, c=Y, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Output')
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_zticks(np.arange(0.0, 1.1, 0.1))

    plt.show()

# AND 게이트의 출력 값 그래프
plot_output_values(X, predictions_and, 'AND Gate')

# OR 게이트의 출력 값 그래프
plot_output_values(X, predictions_or, 'OR Gate')
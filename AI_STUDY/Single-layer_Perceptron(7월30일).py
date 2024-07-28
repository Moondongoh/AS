import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10000):
        # 입력 크기 + 1 (편향) 만큼의 무작위 가중치 초기화
        self.W = np.random.uniform(0.1, 0.9, input_size + 1)
        self.lr = lr
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

    def fit(self, X, d):
        # 학습 데이터에 대해 모델 학습
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)  # 편향 b를 위한 입력 값 추가
                z = self.W.T.dot(x)
                y = self.activation_fn(z)
                e = d[i] - y
                # 가중치 업데이트
                self.W = self.W + self.lr * e * x * y * (1 - y)
                # 가중치의 변화를 기록
                self.history.append(self.W.copy())

# x1과 x2에 대한 입력 값 생성
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

print("\nTraining OR Perceptron:")
or_perceptron.fit(X, d_or)

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

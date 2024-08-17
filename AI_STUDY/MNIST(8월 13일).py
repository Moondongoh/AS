import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 데이터 로딩
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype(int)  # 레이블을 정수형으로 변환

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 데이터 표준화

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 신경망 클래스 정의
class SimpleMLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size, learning_rate=0.001, epochs=10000):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 가중치 초기화 (He 초기화)
        self.weights = []
        self.biases = []
        prev_size = input_size
        for size in hidden_layer_sizes:
            self.weights.append(np.random.randn(prev_size, size) * np.sqrt(2. / prev_size))
            self.biases.append(np.zeros((1, size)))
            prev_size = size
        self.weights.append(np.random.randn(prev_size, output_size) * np.sqrt(2. / prev_size))
        self.biases.append(np.zeros((1, output_size)))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.relu(z)
            self.zs.append(z)
            self.activations.append(a)
        self.output = self.softmax(self.activations[-1])
        return self.output

    def backward(self, X, y):
        m = y.shape[0]
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y] = 1

        dA = self.output - y_one_hot
        for i in reversed(range(len(self.weights))):
            dZ = dA * self.relu_derivative(self.zs[i])
            dW = np.dot(self.activations[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB

    def train(self, X, y):
        for epoch in range(self.epochs):
            self.forward(X)
            self.backward(X, y)
            if (epoch + 1) % 100 == 0:  # 1000개 에포크마다 출력
                loss = -np.mean(np.log(self.output[np.arange(y.shape[0]), y]))
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 신경망 초기화 및 학습
input_size = X_train.shape[1]
hidden_layer_sizes = [100, 300]
output_size = 10
mlp = SimpleMLP(input_size, hidden_layer_sizes, output_size, learning_rate=0.001, epochs=10000)

# 훈련
mlp.train(X_train, y_train)

# 모델 평가
num_samples = 10000
X_test_subset = X_test[:num_samples]
y_test_subset = y_test[:num_samples]

y_pred_subset = mlp.predict(X_test_subset)
accuracy_subset = accuracy_score(y_test_subset, y_pred_subset)
loss = -np.mean(np.log(mlp.forward(X_test_subset)[np.arange(num_samples), y_test_subset]))

print(f'\n테스트셋의 정확도: {accuracy_subset:.4f}')
print(f'테스트셋의 손실율: {loss:.4f}')

# 성공과 실패 개수 출력.
success_count = np.sum(y_pred_subset == y_test_subset)
failure_count = num_samples - success_count
print(f'1만 개 샘플 중 성공 개수: {success_count}')
print(f'1만 개 샘플 중 실패 개수: {failure_count}')

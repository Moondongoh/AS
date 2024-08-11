import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        # 입력층과 첫 번째 은닉층 사이의 가중치 행렬
        self.weights1 = np.random.randn(input_size, hidden_size)
        # 은 편
        self.bias1 = np.zeros((1, hidden_size))
        # 첫 번째 은닉층과 출력층 사이의 가중치 행렬.
        self.weights2 = np.random.randn(hidden_size, output_size)
        # 출 편
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # 미분 값
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, input_data):
        # 첫 번째 은닉층
        # 입력 데이터와 첫 번째 은닉층 가중치의 곱과 편향을 더한 값.
        self.hidden_layer_input = np.dot(input_data, self.weights1) + self.bias1
        # 시그모이드 활성화 함수를 적용한 값.
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        # 출력층
        # 은닉층 출력과 출력층 가중치의 곱과 편향을 더한 값.
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights2) + self.bias2
        # 시그모이드 활성화 함수를 적용한 최종 출력값.
        self.predicted_output = self.sigmoid(self.output_layer_input)
        
        return self.predicted_output
    
    def backward(self, input_data, output_data):
        # 오차 계산
        error = output_data - self.predicted_output
        
        # 출력층의 기울기, 오차에 시그모이드 도함수를 곱한 값
        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)
        
        # 두 번째 은닉층에서의 기울기
        d_weights2 = np.dot(self.hidden_layer_output.T, d_predicted_output)
        d_bias2 = np.sum(d_predicted_output, axis=0, keepdims=True)
        
        # 첫 번째 은닉층에서의 기울기
        d_hidden_layer_output = np.dot(d_predicted_output, self.weights2.T) * self.sigmoid_derivative(self.hidden_layer_output)
        d_weights1 = np.dot(input_data.T, d_hidden_layer_output)
        d_bias1 = np.sum(d_hidden_layer_output, axis=0, keepdims=True)
        
        # 가중치와 편향 업데이트
        self.weights2 += self.learning_rate * d_weights2
        self.bias2 += self.learning_rate * d_bias2
        self.weights1 += self.learning_rate * d_weights1
        self.bias1 += self.learning_rate * d_bias1
    
    def train(self, input_data, output_data, epochs=100000):
        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, output_data)
            if (epoch + 1) % 10000 == 0:
                loss = np.mean(np.square(output_data - self.predicted_output))
                print(f"반복수 {epoch + 1}, 오차: {loss}")
    
    def predict(self, input_data):
        return self.forward(input_data)

# 입력 및 출력 데이터
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 신경망 학습
print("학습:")
perceptron = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=2)
perceptron.train(input_data, output_data, epochs=100000)

# 예측
predictions = perceptron.predict(input_data)
print("예측값:")
print(predictions)

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        np.random.seed(1)
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        self.bias_hidden = np.random.uniform(size=(1, hidden_size))
        self.bias_output = np.random.uniform(size=(1, output_size))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)
        
        return self.predicted_output

    def backward(self, X, y, predicted_output):
        error = y - predicted_output
        d_predicted_output = error * self.sigmoid_derivative(predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)
        
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * self.lr
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.lr
        
        self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.lr
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predicted_output = self.forward(X)
            self.backward(X, y, predicted_output)
            if epoch % 1000 == 0:
                error = np.mean(np.abs(y - predicted_output))
                print(f'Error at epoch {epoch}: {error}')
    
    def predict(self, X):
        return self.forward(X)

# 데이터셋 정의
# 각 패턴을 3x3 격자로 표현한 것
X = np.array([
    [1, 1, 1, 0, 1, 0, 0, 1, 0],  # 첫 번째 패턴
    [1, 1, 1, 1, 0, 0, 1, 1, 1],  # 두 번째 패턴
    [1, 1, 1, 0, 0, 1, 0, 0, 1],  # 세 번째 패턴
])

y = np.array([
    [1, 0, 0],  # 첫 번째 패턴은 클래스 1
    [0, 1, 0],  # 두 번째 패턴은 클래스 2
    [0, 0, 1],  # 세 번째 패턴은 클래스 3
])

# MLP 모델 생성 및 학습
mlp = MLP(input_size=9, hidden_size=5, output_size=3, lr=0.1)
mlp.train(X, y, epochs=10000)

# 테스트 데이터
test_data = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0]])  # 세 번째 패턴과 유사한 데이터
print("Test prediction:", mlp.predict(test_data))

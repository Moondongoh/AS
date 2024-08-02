import numpy as np

class ThreeLayerPerceptron:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.1
        self.bias1 = np.zeros((1, hidden_size1))
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        self.bias2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, hidden_size3) * 0.1
        self.bias3 = np.zeros((1, hidden_size3))
        self.weights4 = np.random.randn(hidden_size3, output_size) * 0.1
        self.bias4 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def linear(self, x):
        return x
    
    def linear_derivative(self, x):
        return np.ones_like(x)
    
    def forward(self, input_data):
        self.hidden_layer1_input = np.dot(input_data, self.weights1) + self.bias1
        self.hidden_layer1_output = self.relu(self.hidden_layer1_input)
        
        self.hidden_layer2_input = np.dot(self.hidden_layer1_output, self.weights2) + self.bias2
        self.hidden_layer2_output = self.relu(self.hidden_layer2_input)
        
        self.hidden_layer3_input = np.dot(self.hidden_layer2_output, self.weights3) + self.bias3
        self.hidden_layer3_output = self.relu(self.hidden_layer3_input)
        
        self.output_layer_input = np.dot(self.hidden_layer3_output, self.weights4) + self.bias4
        self.predicted_output = self.linear(self.output_layer_input)
        
        return self.predicted_output
    
    def backward(self, input_data, output_data):
        error = output_data - self.predicted_output
        
        d_predicted_output = error * self.linear_derivative(self.output_layer_input)
        
        d_weights4 = np.dot(self.hidden_layer3_output.T, d_predicted_output)
        d_bias4 = np.sum(d_predicted_output, axis=0, keepdims=True)
        
        d_hidden_layer3_output = np.dot(d_predicted_output, self.weights4.T) * self.relu_derivative(self.hidden_layer3_input)
        d_weights3 = np.dot(self.hidden_layer2_output.T, d_hidden_layer3_output)
        d_bias3 = np.sum(d_hidden_layer3_output, axis=0, keepdims=True)
        
        d_hidden_layer2_output = np.dot(d_hidden_layer3_output, self.weights3.T) * self.relu_derivative(self.hidden_layer2_input)
        d_weights2 = np.dot(self.hidden_layer1_output.T, d_hidden_layer2_output)
        d_bias2 = np.sum(d_hidden_layer2_output, axis=0, keepdims=True)
        
        d_hidden_layer1_output = np.dot(d_hidden_layer2_output, self.weights2.T) * self.relu_derivative(self.hidden_layer1_input)
        d_weights1 = np.dot(input_data.T, d_hidden_layer1_output)
        d_bias1 = np.sum(d_hidden_layer1_output, axis=0, keepdims=True)
        
        self.weights4 += self.learning_rate * d_weights4
        self.bias4 += self.learning_rate * d_bias4
        self.weights3 += self.learning_rate * d_weights3
        self.bias3 += self.learning_rate * d_bias3
        self.weights2 += self.learning_rate * d_weights2
        self.bias2 += self.learning_rate * d_bias2
        self.weights1 += self.learning_rate * d_weights1
        self.bias1 += self.learning_rate * d_bias1

        # Return errors for each hidden layer
        return d_hidden_layer1_output, d_hidden_layer2_output, d_hidden_layer3_output
    
    def train(self, input_data, output_data, epochs=5000):
        for epoch in range(epochs):
            self.forward(input_data)
            d_hidden_layer1_output, d_hidden_layer2_output, d_hidden_layer3_output = self.backward(input_data, output_data)
            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(output_data - self.predicted_output))
                print(f"Epoch {epoch + 1}, Loss: {loss}")
    
    def predict(self, input_data):
        return self.forward(input_data)

# 입력 및 출력 데이터 정의
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 신경망 초기화 및 학습
perceptron = ThreeLayerPerceptron(input_size=2, hidden_size1=3, hidden_size2=3, hidden_size3=3, output_size=2)
perceptron.train(input_data, output_data, epochs=5000)

# 예측
predictions = perceptron.predict(input_data)
print("Predictions:")
print(predictions)

# 은닉층의 에러값 출력
_, _, hidden_layer_errors = perceptron.backward(input_data, output_data)
print("Hidden layer errors:")
print(hidden_layer_errors)

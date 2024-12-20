# 2 3 1
import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, input_data):
        self.hidden_layer_input = np.dot(input_data, self.weights1) + self.bias1
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights2) + self.bias2
        self.predicted_output = self.sigmoid(self.output_layer_input)
        
        return self.predicted_output
    
    def backward(self, input_data, output_data):
        error = output_data - self.predicted_output
        
        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)
        
        d_weights2 = np.dot(self.hidden_layer_output.T, d_predicted_output)
        d_bias2 = np.sum(d_predicted_output, axis=0, keepdims=True)
        
        d_hidden_layer_output = np.dot(d_predicted_output, self.weights2.T) * self.relu_derivative(self.hidden_layer_input)
        d_weights1 = np.dot(input_data.T, d_hidden_layer_output)
        d_bias1 = np.sum(d_hidden_layer_output, axis=0, keepdims=True)
        
        self.weights2 += self.learning_rate * d_weights2
        self.bias2 += self.learning_rate * d_bias2
        self.weights1 += self.learning_rate * d_weights1
        self.bias1 += self.learning_rate * d_bias1
    
    def train(self, input_data, output_data, epochs=10000):
        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, output_data)
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(output_data - self.predicted_output))
                print(f"Epoch {epoch + 1}, Loss: {loss}")
    
    def predict(self, input_data):
        return self.forward(input_data)

# AND 게이트 입력 및 출력 데이터
input_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data_and = np.array([[0], [0], [0], [1]])

# OR 게이트 입력 및 출력 데이터
input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data_or = np.array([[0], [1], [1], [1]])

# XOR 게이트 입력 및 출력 데이터
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data_xor = np.array([[0], [1], [1], [0]])

# AND 게이트 학습
print("Training AND gate:")
perceptron_and = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=1)
perceptron_and.train(input_data_and, output_data_and, epochs=10000)

# AND 게이트 예측
predictions_and = perceptron_and.predict(input_data_and)
print("AND gate predictions:")
print(predictions_and)

# OR 게이트 학습
print("\nTraining OR gate:")
perceptron_or = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=1)
perceptron_or.train(input_data_or, output_data_or, epochs=10000)

# OR 게이트 예측
predictions_or = perceptron_or.predict(input_data_or)
print("OR gate predictions:")
print(predictions_or)

# XOR 게이트 학습
print("\nTraining XOR gate:")
perceptron_xor = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=1)
perceptron_xor.train(input_data_xor, output_data_xor, epochs=10000)

# XOR 게이트 예측
predictions_xor = perceptron_xor.predict(input_data_xor)
print("XOR gate predictions:")
print(predictions_xor)

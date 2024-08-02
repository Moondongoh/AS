import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 가중치 초기화
        self.w1 = np.random.rand(input_size, hidden_layer1_size)
        self.w2 = np.random.rand(hidden_layer1_size, hidden_layer2_size)
        self.w3 = np.random.rand(hidden_layer2_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        self.z1 = np.dot(inputs, self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w3)
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def backward_propagation(self, inputs, outputs):
        output_error = outputs - self.a3
        output_delta = output_error * self.sigmoid_derivative(self.a3)

        hidden2_error = output_delta.dot(self.w3.T)
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.a2)

        hidden1_error = hidden2_delta.dot(self.w2.T)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.a1)

        self.w3 += self.a2.T.dot(output_delta) * self.learning_rate
        self.w2 += self.a1.T.dot(hidden2_delta) * self.learning_rate
        self.w1 += inputs.T.dot(hidden1_delta) * self.learning_rate

        return hidden1_error, hidden2_error
    
    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            hidden1_error, hidden2_error = self.backward_propagation(inputs, outputs)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(outputs - self.a3))
                print(f'Epoch {epoch} Loss: {loss}')
        
        return hidden1_error, hidden2_error

# Define constants
INPUT_SIZE = 2
HIDDEN_LAYER1_SIZE = 2
HIDDEN_LAYER2_SIZE = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 10000

# XOR 데이터셋
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
outputs = np.array([[0], [1], [1], [0]])

# 모델 초기화
mlp = MultiLayerPerceptron(INPUT_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, OUTPUT_SIZE, LEARNING_RATE)

# 학습 실행
hidden1_error, hidden2_error = mlp.train(inputs, outputs, EPOCHS)

print("최종 가중치:")
print("w1:\n", mlp.w1)
print("w2:\n", mlp.w2)
print("w3:\n", mlp.w3)

print("은닉층 1의 에러값:")
print(hidden1_error)
print("은닉층 2의 에러값:")
print(hidden2_error)

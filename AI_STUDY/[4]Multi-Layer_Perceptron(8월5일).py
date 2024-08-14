import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.w1 = np.random.rand(input_size, hidden_layer_size)
        self.w2 = np.random.rand(hidden_layer_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        self.z1 = np.zeros((inputs.shape[0], self.hidden_layer_size))
        for i in range(inputs.shape[0]):
            for j in range(self.hidden_layer_size):
                self.z1[i][j] = sum(inputs[i][k] * self.w1[k][j] for k in range(self.input_size))
        
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.zeros((self.a1.shape[0], self.output_size))
        for i in range(self.a1.shape[0]):
            for j in range(self.output_size):
                self.z2[i][j] = sum(self.a1[i][k] * self.w2[k][j] for k in range(self.hidden_layer_size))
        
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward_propagation(self, inputs, outputs):
        output_error = outputs - self.a2
        output_delta = output_error * self.sigmoid_derivative(self.a2)
        
        hidden_error = np.zeros((output_delta.shape[0], self.hidden_layer_size))
        for i in range(output_delta.shape[0]):
            for j in range(self.hidden_layer_size):
                hidden_error[i][j] = sum(output_delta[i][k] * self.w2[j][k] for k in range(self.output_size))
        
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        for i in range(self.w2.shape[0]):
            for j in range(self.w2.shape[1]):
                for k in range(self.a1.shape[0]):
                    self.w2[i][j] += self.a1[k][i] * output_delta[k][j] * self.learning_rate
        
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                for k in range(inputs.shape[0]):
                    self.w1[i][j] += inputs[k][i] * hidden_delta[k][j] * self.learning_rate
        
        return hidden_error
    
    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            hidden_error = self.backward_propagation(inputs, outputs)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(outputs - self.a2))
                print(f'Epoch {epoch} Loss: {loss}')
        
        return hidden_error

    def predict(self, inputs):
        return self.forward_propagation(inputs)

# Define constants
INPUT_SIZE = 2
HIDDEN_LAYER_SIZE = 3
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
mlp = MultiLayerPerceptron(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE, LEARNING_RATE)

# 학습 실행
hidden_error = mlp.train(inputs, outputs, EPOCHS)

print("최종 가중치:")
print("w1:\n", mlp.w1)
print("w2:\n", mlp.w2)

print("은닉층의 에러값:")
print(hidden_error)

# 예측값 계산
predictions = mlp.predict(inputs)
print("예측값:")
print(predictions)

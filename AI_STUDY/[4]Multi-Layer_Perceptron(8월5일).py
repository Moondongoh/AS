import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate):
        # 입력, 은닉, 출력층의 크기를 상수로 받고 학습률을 선정하기 위해서
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 가중치 초기화 (초기값은 랜덤하게 받음)
        self.w1 = np.random.rand(input_size, hidden_layer_size)
        self.w2 = np.random.rand(hidden_layer_size, output_size)
        
        # 시그모이드 함수
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
        # 시그모이드 도함수 (미분 역전파에 사용)
        # 은닉층 오차 = 출력층 기울기 * 가중치 * 활성 함수의 도함수
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        # 입력 데이터와 첫 번째 가중치 행렬의 곱
        self.z1 = np.dot(inputs, self.w1)
        # 첫 번째 은닉층의 활성화
        self.a1 = self.sigmoid(self.z1)
        # 은닉층과 두 번째 가중치 행렬의 곱
        self.z2 = np.dot(self.a1, self.w2)
        # 출력층의 활성화 
        self.a2 = self.sigmoid(self.z2)
        # 출력층의 활성화 값
        return self.a2
    
    def backward_propagation(self, inputs, outputs):
        # 출력층 오차 구함
        output_error = outputs - self.a2
        # 기울기? 출력층 에러에 도함수를 곱하면 뉴런의 기울기가 나오는데 이걸로 가중치 업데이트
        output_delta = output_error * self.sigmoid_derivative(self.a2)

        # 은닉층 오차 구함
        hidden_error = output_delta.dot(self.w2.T)
        # 은닉층의 각 뉴런의 기울기
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.w2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.w1 += inputs.T.dot(hidden_delta) * self.learning_rate

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

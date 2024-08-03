# import numpy as np

# class Perceptron:
#     def __init__(self, input_size, lr=1, epochs=10):
#         self.W = np.zeros(input_size + 1)
#         self.lr = lr
#         self.epochs = epochs

#     def activation_fn(self, x):
#         return 1 if x >= 0 else 0

#     def predict(self, x):
#         z = self.W.T.dot(np.insert(x, 0, 1))
#         a = self.activation_fn(z)
#         return a

#     def fit(self, X, d):
#         for epoch in range(self.epochs):
#             print(f"Epoch {epoch + 1}")
#             for i in range(d.shape[0]):
#                 x = np.insert(X[i], 0, 1)  # 편향 b를 위한 입력 값 추가
#                 y = self.predict(X[i])
#                 e = d[i] - y
#                 self.W = self.W + self.lr * e * x
#                 print(f"  Sample {X[i]} -> Target: {d[i]}, Predicted: {y}, Error: {e}")
#                 print(f"  Updated Weights: {self.W}")
#             print()

# # AND gate
# X_and = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# d_and = np.array([0, 0, 0, 1])

# # OR gate
# X_or = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# d_or = np.array([0, 1, 1, 1])

# # Create perceptron models
# and_perceptron = Perceptron(input_size=2, epochs=10)
# or_perceptron = Perceptron(input_size=2, epochs=10)

# # Train the models
# print("Training AND Perceptron:")
# and_perceptron.fit(X_and, d_and)

# print("\nTraining OR Perceptron:")
# or_perceptron.fit(X_or, d_or)

# # Test the models
# print("AND Perceptron Results:")
# for x in X_and:
#     print(f"{x} -> {and_perceptron.predict(x)}")

# print("\nOR Perceptron Results:")
# for x in X_or:
#     print(f"{x} -> {or_perceptron.predict(x)}")

# ------------------------------------------------------------

import numpy as np

# AND 게이트 구현
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = 0.9
    tmp = np.sum(x * w)
    if tmp <= bias:
        return 0
    if tmp > bias:
        return 1


# OR 게이트 구현
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = 0.1
    tmp = np.sum(x * w)
    if tmp <= bias:
        return 0
    if tmp >= bias:
        return 1

# 테스트
test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("AND 게이트 결과:")
for x1, x2 in test_cases:
    print(f"AND({x1}, {x2}) = {AND(x1, x2)}")

print("\nOR 게이트 결과:")
for x1, x2 in test_cases:
    print(f"OR({x1}, {x2}) = {OR(x1, x2)}")

# ------------------------------------------------------------

# import numpy as np

# # 퍼셉트론 함수 정의
# def perceptron(x1, x2, weights, bias):
#     x = np.array([x1, x2])
#     w = np.array(weights)
#     z = np.sum(x * w) + bias
#     if z < 0:
#         return 0
#     if z >= 0:
#         return 1
#     #return 1 if z > 0 else 0

# # 테스트 케이스
# test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

# # AND 게이트의 원하는 출력
# desired_outputs_and = {
#     (0, 0): 0,
#     (0, 1): 0,
#     (1, 0): 0,
#     (1, 1): 1
# }

# # OR 게이트의 원하는 출력
# desired_outputs_or = {
#     (0, 0): 0,
#     (0, 1): 1,
#     (1, 0): 1,
#     (1, 1): 1
# }

# # 랜덤 가중치와 편향 초기화
# np.random.seed(42)  # 재현성을 위해 시드 설정
# weights = np.random.uniform(0, 1, size=(2,))  # -1에서 1 사이의 랜덤 값으로 가중치 초기화
# bias_and = np.random.uniform(0, 1)  # AND 게이트를 위한 랜덤 편향 초기화
# bias_or = np.random.uniform(0, 1)   # OR 게이트를 위한 랜덤 편향 초기화

# # 퍼셉트론 결과 출력
# def print_results(name, weights, bias, desired_outputs):
#     print("\n")
#     print(f"{name} 게이트 결과 (가중치: {weights}, 편향: {bias}):")
#     results = {}
#     for x1, x2 in test_cases:
#         output = perceptron(x1, x2, weights, bias)
#         results[(x1, x2)] = output
#         print(f"{name}({x1}, {x2}) = {output}")
#     # print(f"\n{name} 게이트의 원하는 출력:")
#     # for (x1, x2), output in desired_outputs.items():
#     #     print(f"{name}({x1}, {x2}) = {output}")
#     return results

# # 현재 가중치와 편향으로 AND 게이트 및 OR 게이트의 동작을 확인
# print_results("F_AND", weights, bias_and, desired_outputs_and)
# print_results("F_OR", weights, bias_or, desired_outputs_or)

# # 편향을 조정하여 원하는 결과를 얻는 방법
# def find_optimal_bias(weights, desired_outputs):
#     for bias in np.arange(-1, 1, 0.1):
#         results = { (x1, x2): perceptron(x1, x2, weights, bias) for (x1, x2) in test_cases }
        
#         # 현재 편향 값과 그 결과를 출력
#         print("\n")
#         print(f"편향: {bias}")
#         for (x1, x2), output in results.items():
#             print(f"  입력({x1}, {x2}) -> 출력: {output}")
        
#         if all(results[(x1, x2)] == desired_outputs[(x1, x2)] for (x1, x2) in test_cases):
#             print(f"최적의 편향을 찾았습니다: {bias}")
#             return bias
            
#         print("-" * 30)
    
#     print("적절한 편향을 찾을 수 없습니다.")
#     return None

# # 최적의 편향 찾기
# optimal_bias_and = find_optimal_bias(weights, desired_outputs_and)
# optimal_bias_or = find_optimal_bias(weights, desired_outputs_or)

# print(f"\nAND 게이트를 구현하기 위한 최적의 편향: {optimal_bias_and}")
# print(f"OR 게이트를 구현하기 위한 최적의 편향: {optimal_bias_or}")

# # 최적의 편향으로 다시 테스트
# if optimal_bias_and is not None:
#     print_results("AND (최적 편향)", weights, optimal_bias_and, desired_outputs_and)
# else:
#     print("최적의 편향을 찾을 수 없습니다.")

# if optimal_bias_or is not None:
#     print_results("OR (최적 편향)", weights, optimal_bias_or, desired_outputs_or)
# else:
#     print("최적의 편향을 찾을 수 없습니다.")

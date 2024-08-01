import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10000):
        self.W = np.random.uniform(0.1, 0.9, input_size + 1)
        self.lr = lr
        self.epochs = epochs
        self.history = []

    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        z = self.W.T.dot(np.insert(x, 0, 1))
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)  # 편향 b를 위한 입력 값 추가
                z = self.W.T.dot(x)
                y = self.activation_fn(z)
                e = d[i] - y
                self.W = self.W + self.lr * e * x * y * (1 - y)
                self.history.append(self.W.copy())

# Generate input values for x1 and x2
X = np.array([[x1, x2] for x1 in np.arange(0.1, 1.0, 0.1) for x2 in np.arange(0.1, 1.0, 0.1)])

# Generate desired output values for AND and OR gates
d_and = np.array([int(x1 > 0.5 and x2 > 0.5) for x1, x2 in X])
d_or = np.array([int(x1 > 0.1 or x2 > 0.1) for x1, x2 in X])

# Create perceptron models
and_perceptron = Perceptron(input_size=2, epochs=10000)
or_perceptron = Perceptron(input_size=2, epochs=10000)

# Train the models
print("Training AND Perceptron:")
and_perceptron.fit(X, d_and)

print("\nTraining OR Perceptron:")
or_perceptron.fit(X, d_or)

# Generate predictions
predictions_and = np.array([and_perceptron.predict(x) for x in X])
predictions_or = np.array([or_perceptron.predict(x) for x in X])

# Plotting the output values
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

    plt.show()

# Plot the output values for the AND gate
plot_output_values(X, predictions_and, 'AND Gate')

# Plot the output values for the OR gate
plot_output_values(X, predictions_or, 'OR Gate')

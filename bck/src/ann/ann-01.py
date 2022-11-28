import numpy as np

from activation_functions.sigmoid_function import sigmoid_function

x = np.array([2, 5, 7])
w = np.array([0.4, 0.2, 0.7])
Z = np.sum(x * w)
y = sigmoid_function(Z)

print(x)
print(w)
print(Z)
print(y)

print('-' * 60)

X = np.array([1, 3, 5])
B = np.array([0.2, 0.3])
W = np.array([[0.3, 0.5], [0.4, 0.2], [0.7, 0.3]])

print("X shape:", X.shape)
print("B shape:", B.shape)
print("W shape:", W.shape)

T = np.dot(X, W)
Z = T + B
print(X)
print(B)
print(W)
print(T)
print(Z)

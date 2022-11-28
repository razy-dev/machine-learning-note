import random
from typing import Final, Generator

import numpy as np
from numpy import ndarray, array

INPUT_SIZE: Final = 2  # y = wx + wx + ... + b
DATA_SIZE: Final = 5


def model(x, w, b):
    return np.dot(x, w) + b


def loss(y: ndarray, y_hat: ndarray):
    return (y.reshape(y_hat.shape) - y_hat) ** 2 / 2


def optimizer(params, lr, size):
    pass


# Create Data
def dataset(w: array, b, noise=0.01) -> (ndarray, ndarray):
    features = np.random.normal(scale=1, size=(DATA_SIZE, INPUT_SIZE))
    label = model(features, w, b) + np.random.normal(scale=noise, size=(DATA_SIZE,))
    return features, label


def mini_batch(features: ndarray, label: ndarray, size: int = 2, shuffle=True) -> Generator:
    total = len(features)
    indices = list(range(total))
    shuffle and random.shuffle(indices)
    for i in range(0, total, size):
        end = min(i + size, total)
        yield features.take(indices[i:end], axis=0), label.take(indices[i:end])


lr = 0.01
epochs = 5
net = model
cost = loss

w = np.random.normal(scale=0.01, size=(INPUT_SIZE, 1))
w.attach_grad()
b = np.zeros(size=(1,))
b.attach_grad()
for epoch in range(epochs):
    for x, y in mini_batch(*dataset(np.array([1, 1]), 1), 2, shuffle=False):
        pass

# print(f)
# print(f[:, 0])
# print(f[:, -1])
# print(l)
#
# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(f[:, 0], f[:, -1], l)
# # plt.scatter(f[:, 1], l)
# plt.show()

import numpy as np
from matplotlib import pyplot as plt


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def gh_sigmoid_function():
    x = np.arange(-10.0, 10.0, 0.1)
    y = sigmoid_function(x)

    plt.figure(figsize=(8, 8))
    plt.title("Sigmoid Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.xlim(-10, 10)
    plt.yticks([0.0, 0.5, 1.0])
    plt.gca().yaxis.grid(True)
    plt.axvline(0.0, c="black")

    plt.show()


if __name__ == '__main__':
    gh_sigmoid_function()

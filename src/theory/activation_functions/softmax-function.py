import numpy as np
from matplotlib import pyplot as plt


def softmax_function(x):
    max_x = np.max(x)
    min_x = np.min(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def gh_softmax_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = softmax_function(x)

    plt.figure(figsize=(8, 8))
    plt.title("Softmax Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(0, 0.1)

    plt.show()


if __name__ == '__main__':
    gh_softmax_function()

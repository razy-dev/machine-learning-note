import numpy as np
from matplotlib import pyplot as plt


def gh_exp_function():
    x = np.arange(-5, 5, 0.1)
    y = np.exp(x)

    plt.figure(figsize=(8, 6))  # 캔버스 생성
    plt.title("Exponential Function", fontsize=25)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15, rotation=0)

    plt.plot(x, y)

    plt.show()


def softmax_function(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def gh_softmax_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = softmax_function(x)

    plt.figure(figsize=(8, 8))
    plt.title("Sigmoid Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(0, 0.1)

    plt.show()


if __name__ == '__main__':
    # gh_exp_function()
    gh_softmax_function()

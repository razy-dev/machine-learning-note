import matplotlib.pyplot as plt
import numpy as np


def linear_function(x, k):
    return k * x


def gh_linear_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = linear_function(x, 0.7)

    plt.figure(figsize=(8, 8))  # 캔버스 생성
    plt.title("Step Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(-1, 4)
    plt.xlim(-1, 4)
    plt.axhline(c="black")
    plt.axvline(c="black")

    plt.show()


if __name__ == '__main__':
    gh_linear_function()

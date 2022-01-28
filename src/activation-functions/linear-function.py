import matplotlib.pyplot as plt
import numpy as np


def step_function(x, k):
    return k * x


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x, 0.7)

    fig = plt.figure(figsize=(8, 8))  # 캔버스 생성
    plt.title("Step Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(-1, 4)
    plt.xlim(-1, 4)
    plt.axhline(c="black")
    plt.axvline(c="black")

    plt.show()

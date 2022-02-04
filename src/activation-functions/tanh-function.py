import matplotlib.pyplot as plt
import numpy as np


def tanh(x):
    p_exp_x = np.exp(x)
    n_exp_x = np.exp(-x)
    return (p_exp_x - n_exp_x) / (p_exp_x + n_exp_x)


def gh_tanh():
    x = np.arange(-5.0, 5.0, 0.1)
    y = tanh(x)

    plt.figure(figsize=(8, 8))  # 캔버스 생성
    plt.title("Hyperbolic Tangent", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.yticks([-1.0, 0.0, 1.0])
    plt.axvline(0.0, color='k')
    plt.gca().yaxis.grid(True)

    plt.show()


if __name__ == '__main__':
    gh_tanh()

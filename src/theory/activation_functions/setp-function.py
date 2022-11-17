import matplotlib.pyplot as plt
import numpy as np


def step_function(x):
    return (x > 0).astype(int)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)

    fig = plt.figure(figsize=(8, 8))
    plt.title("Step Function", fontsize=30)
    plt.ylabel('y', fontsize=20, rotation=0)
    plt.xlabel('x', fontsize=20)

    plt.plot(x, y)
    plt.ylim(-0.5, 1.5)
    plt.xlim(-5, 5)

    plt.show()

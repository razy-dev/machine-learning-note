import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

# tag::data[]
MAX = 100
x = np.linspace(0, 10, MAX, endpoint=True)
y = 10 * x + np.random.randint(-10, 10, MAX)


# end::data[]


# tag::hypothesis[]
# y = w * x + b
def hypothesis(x):
    return w * x + b


# end::hypothesis[]

# tag::cost[]
# cost = 1/n(sum((y - y_pred)^2) = 1/n(sum((y - (w * x + b))^2)
def cost(y: ndarray, x: ndarray):
    return np.sum((y - hypothesis(x)) ** 2) / len(y)


# end::cost[]

# tag::dcost_dw[]
# dw = -2w/m(sum((y - (w * x +b))) = -2w/m(sum(y-y_pred))
def dcost_dw(y: ndarray, x: ndarray):
    return (-2 / len(y)) * np.sum(x * (y - hypothesis(x)))


# end::dcost_dw[]

# tag::dcost_db[]
# db = -2/m(sum((y - (w * x +b))) = = -2/m(sum(y-y_pred))
def dcost_db(y: ndarray, x: ndarray):
    return (-2 / len(y)) * np.sum(y - hypothesis(x))


# end::dcost_db[]


# tag::hyperparameter[]
w = 5
b = 1

# end::hyperparameter[]

# tag::train[]
costs = []
lr = 0.001
for e in range(100):
    costs.append(cost(y, x))
    w = w - lr * dcost_dw(y, x)
    b = b - lr * dcost_db(y, x)

    plt.subplot(211)
    if not e % 10: plt.plot(x, hypothesis(x), '--y', label=f"epoch {e}")

plt.subplot(211)
plt.plot(x, y, 'oc')
plt.plot(x, hypothesis(x), 'r')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.subplot(212)
plt.plot(costs)
plt.xlabel('epoch')
plt.ylabel('cost')

plt.show()
# end::train[]

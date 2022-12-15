# tag::preset[]
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

matplotlib.use('MacOSX')  # 'module://backend_interagg', 'GTK3Agg', 'GTK3Cairo' ...
# end::preset[]

# tag::dataset[]
BATCH_SIZE = 100

np.random.seed(1)  # 난수 고정
x = np.linspace(0, 10, BATCH_SIZE, endpoint=False)
y = 10 * x + np.random.randint(-10, 10, BATCH_SIZE)


# Plot
# plt.scatter(x, y)
# plt.show()
# end::dataset[]

# tag::hypothesis[]
def hypothesis(x):
    return w * x + b


# end::hypothesis[]

# tag::cost[]
def cost(input: ndarray, target: ndarray):
    return np.sum((input - target) ** 2) / len(y)


# end::cost[]

# tag::dw[]
def dcost_dw(input: ndarray, target: ndarray, x: ndarray):
    return (-2 / len(target)) * np.sum(x * (target - input))


# end::dw[]

# tag::db[]
def dcost_db(input: ndarray, target: ndarray):
    return (-2 / len(target)) * np.sum(target - input)


# end::db[]

# tag::optimizer[]

# end::optimizer[]

# tag::training[]
# tag::parameters[]
w = 5
b = 1

# end::parameters[]
costs = []
lr = 0.001
for e in range(100):
    y_pred = hypothesis(x)
    costs.append(cost(y_pred, y))
    w = w - lr * dcost_dw(y_pred, y, x)
    b = b - lr * dcost_db(y_pred, y)

    if not e % 10:
        print(f"{e:4} : cost = {costs[-1]} H(x) = {w}x + {b}")
        plt.plot(x, hypothesis(x), '--y', label=f"epoch {e}")
# end::training[]

# plt.subplot(211)
# plt.plot(x, y, 'oc')
# plt.plot(x, hypothesis(x), 'r')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
#
# plt.subplot(212)
# plt.plot(costs)
# plt.xlabel('epoch')
# plt.ylabel('cost')

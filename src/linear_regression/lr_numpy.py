# tag::preset[]
import matplotlib
import numpy as np
from numpy import ndarray

np.random.seed(1)  # 난수 고정
matplotlib.use('MacOSX')  # 'module://backend_interagg', 'GTK3Agg', 'GTK3Cairo' ...
# end::preset[]

# tag::dataset[]
BATCH_SIZE = 100


def build_dataset(size: int, batch_size: int = None, train_rate: float = 0.7) -> tuple:
    x = np.linspace(0, 10, size)
    y = 10 * x + np.random.randint(-2, 2, size)
    train_len = round(size * train_rate)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = build_dataset(BATCH_SIZE)


# Plot
# plt.scatter(x_train, y_train)
# plt.show()
# end::dataset[]

# tag::hypothesis[]
def hypothesis(x):
    return w * x + b


# end::hypothesis[]

# tag::cost[]
def criterion(input: ndarray, target: ndarray):
    return np.sum((target - input) ** 2) / len(input)


# end::cost[]

# tag::dw[]
def w_grad(input: ndarray, target: ndarray, x: ndarray):
    return (-2 / len(target)) * np.sum(x * (target - input))


# end::dw[]

# tag::db[]
def b_grad(input: ndarray, target: ndarray):
    return (-2 / len(target)) * np.sum(target - input)


# end::db[]

# tag::optimizer[]

# end::optimizer[]

# tag::training[]
# tag::parameters[]
w = 5
b = 1

# end::parameters[]
print("\nTraining the model")
costs = []
lr = 0.01
for e in range(1000):
    y_pred = hypothesis(x_train)
    costs.append(criterion(y_pred, y_train))
    w = w - lr * w_grad(y_pred, y_train, x_train)
    b = b - lr * b_grad(y_pred, y_train)

    if not e % 100:
        print(f"{e:4} : cost = {costs[-1]:>4.8} H(x) = {w:>4.5}x + {b:>4.5}")
        # plt.plot(x, hypothesis(x), '--y', label=f"epoch {e}")
# end::training[]

# tag::test[]
print("\nValidate the model")
for i in range(len(x_test)):
    pred = hypothesis(x_test[i])
    print(f"input = {x_test[i]:.5} | output = {pred:.5} | real = {y_test[i]:.5} | loss = {100 * (pred - y_test[i]) / y_test[i]:.3}%")
# end::test[]

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

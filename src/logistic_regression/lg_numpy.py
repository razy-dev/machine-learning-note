# tag::preset[]
import matplotlib
import numpy as np
from numpy import ndarray

np.random.seed(1)  # 난수 고정
matplotlib.use('MacOSX')  # 'module://backend_interagg', 'GTK3Agg', 'GTK3Cairo' ...
# end::preset[]

# tag::dataset[]
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)


def and_dataset() -> tuple:
    x = X
    y = np.array([0, 0, 0, 1], dtype=np.float)
    return x, y, x, y


def or_dataset() -> tuple:
    x = X
    y = np.array([0, 1, 1, 1], dtype=np.float)
    return x, y, x, y


def xor_dataset() -> tuple:
    x = X
    y = np.array([0, 1, 1, 0], dtype=np.float)
    return x, y, x, y


# Plot
# plt.scatter(x_train, y_train)
# plt.show()
# end::dataset[]

# tag::hypothesis[]
def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def hypothesis(x0, x1):
    return sigmoid(w0 * x0 + w1 * x1 + b)


# end::hypothesis[]

# tag::cost[]
def loss(prediction: ndarray, target: ndarray):
    case_one = -target * np.log(prediction)
    case_zero = -(1 - target) * np.log(prediction)
    return case_one + case_zero


def criterion(prediction: ndarray, target: ndarray):
    return np.sum(loss(prediction, target)) / len(prediction)


# end::cost[]

# tag::dw[]


def w_grad(prediction: ndarray, target: ndarray, x: ndarray):
    return np.sum(x * (1target - input))


# end::dw[]

# tag::db[]
def b_grad(input: ndarray, target: ndarray):
    return (-2 / len(target)) * np.sum(target - input)


# end::db[]

# tag::optimizer[]

# end::optimizer[]

# tag::training[]
# tag::parameters[]
w0 = 1
w1 = 1
b = 0

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

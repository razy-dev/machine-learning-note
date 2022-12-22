# tag::preset[]
import matplotlib
import torch
from torch import Tensor

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing
matplotlib.use('MacOSX')  # 'module://backend_interagg', 'GTK3Agg', 'GTK3Cairo' ...
# end::preset[]

# tag::dataset[]
BATCH_SIZE = 100


def build_dataset(size: int, batch_size: int = None, train_rate: float = 0.7) -> tuple:
    x = torch.linspace(0, 10, steps=size, dtype=float)
    y = 10 * x + torch.randint(-2, 2, (size,))
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
def hypothesis(x) -> Tensor:
    return w * x + b


# end::hypothesis[]

# tag::cost[]
def criterion(input: Tensor, target: Tensor) -> Tensor:
    return torch.sum((target - input) ** 2) / len(input)


# end::cost[]


# tag::optimizer[]
# end::optimizer[]

# tag::training[]
# tag::parameters[]
w = torch.tensor(5., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
# end::parameters[]

print("\nTraining the model")
costs = []
lr = 0.01
for e in range(1000):
    y_pred = hypothesis(x_train)
    cost = criterion(y_pred, y_train)
    costs.append(cost)

    cost.backward()  # Autograd
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad = None  # grad 초기화
        b.grad = None  # grad 초기화

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

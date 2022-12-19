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
class LinearRegession(torch.nn.Module):
    w: Tensor
    b: Tensor

    def __init__(self, w: float = 5., b: float = 1.):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w))
        self.b = torch.nn.Parameter(torch.tensor(b))

    def forward(self, x):
        return self.w * x + self.b


model = LinearRegession()
# end::hypothesis[]

# tag::cost[]
criterion = torch.nn.MSELoss(reduction='mean')

# end::cost[]


# tag::optimizer[]

# end::optimizer[]

# tag::training[]
# tag::parameters[]
# end::parameters[]

print("\nTraining the model")
costs = []
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for e in range(1000):
    y_pred = model(x_train)
    cost = criterion(y_pred, y_train)
    costs.append(cost)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if not e % 100:
        print(f"{e:4} : cost = {costs[-1]:>4.8} H(x) = {model.w.data:>4.5}x + {model.b.data:>4.5}")
        # plt.plot(x, hypothesis(x), '--y', label=f"epoch {e}")
# end::training[]

# tag::test[]
print("\nValidate the model")
model.eval()
with torch.no_grad():
    for i in range(len(x_test)):
        predication = model(x_test[i])
        print(f"input = {x_test[i]:.5} | output = {predication:.5} | real = {y_test[i]:.5} | loss = {100 * (predication - y_test[i]) / y_test[i]:.3}%")
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

# tag::preset[]
import matplotlib
import torch
import torch.nn as nn

from torch import Tensor

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing
matplotlib.use('MacOSX')  # 'module://backend_interagg', 'GTK3Agg', 'GTK3Cairo' ...


# end::preset[]

# tag::dataset[]
def dataset(logic: str) -> tuple:
    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    if "AND" == (logic or '').upper():
        y = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(dim=1)
    elif "OR" == (logic or '').upper():
        y = torch.FloatTensor([0, 1, 1, 1]).unsqueeze(dim=1)
    elif "NAND" == (logic or '').upper():
        y = torch.FloatTensor([1, 1, 1, 0]).unsqueeze(dim=1)
    elif "XOR" == (logic or '').upper():
        y = torch.FloatTensor([0, 1, 1, 0]).unsqueeze(dim=1)
    return x, y, x, y


# Plot
# plt.scatter(x_train, y_train)
# plt.show()
# end::dataset[]

# tag::hypothesis[]
class LogisticRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, ):
        super(LogisticRegression, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


model = LogisticRegression(2, 1)
# end::hypothesis[]

# tag::cost[]
criterion = nn.BCELoss()

# end::cost[]

# tag::optimizer[]
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# end::optimizer[]

# tag::training[]
# tag::parameters[]
params = model.named_parameters()
for param in params:
    print(param)


# end::parameters[]
def train(x_train: Tensor, y_train: Tensor, epochs: int = 1000, threshold: float = 0.3):
    print("Training the model")
    costs = []
    priod = round(epochs / 5)
    for e in range(epochs + 1):
        y_pred = model(x_train)
        cost = criterion(y_pred, y_train)
        costs.append(cost)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (not e % priod) or e == epochs or cost < threshold:
            print(f"{e:4} : cost = {costs[-1]:>4.8}")
            # plt.plot(x, hypothesis(x), '--y', label=f"epoch {e}")

        if cost < threshold: break


# end::training[]

# tag::test[]
def validate(x_test, y_test):
    print("Validate the model")
    model.eval()
    with torch.no_grad():
        for i in range(len(x_test)):
            pred = model(x_test[i]) >= 0.5
            print(f"input = {x_test[i]} | output = {pred} | real = {y_test[i]}")


# end::test[]

# tag::run[]
def runner(logic: str, epochs: int = 1000):
    print(f"\nFor '{(logic or '').upper()}'")
    x_train, y_train, x_test, y_test = dataset(logic)
    train(x_train, y_train, epochs)
    validate(x_test, y_test)


runner('and', 10000)
runner('or', 10000)
runner('nand', 10000)
runner('xor', 10000)
# end::run[]

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

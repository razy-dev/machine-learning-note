import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim

from engine.dataset import dataset, load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

time_steps = 64
features = 1
hidden_size = 1
output_size = 1

x_train, y_train, x_test, y_test, scaler = dataset(
    load(size=1000),
    window_size=time_steps,
    features=features,
    output_size=output_size,
    normalizer='gradient'
)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


class NeuralNetwork(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NeuralNetwork, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=time_steps, dropout=0.25)
        self.lin = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        o, _ = self.rnn(x)
        z = self.lin(o[-1])
        return z  # self.softmax(z)


model = NeuralNetwork(3, 3, 3)
criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()  # couse fo is nn.LogSoftmax
optimizer = optim.Adam(model.parameters(), 0.001)

print("\nTraining ...")
for e in range(100):
    for i, x in enumerate(x_train):
        z: Tensor = model(x)
        y: Tensor = y_train[i].view(-1)
        cost: Tensor = criterion(z, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    print(e, i, z.data.numpy(), y.data.numpy(), cost.item())

outputs = []
rigits = 0
with torch.no_grad():
    for i, x in enumerate(x_test):
        o = model(x)
        z = np.eye(3)[o.data.numpy().argmax()]
        y = y_test.squeeze()[i].data.numpy()
        t = z.argmax() == y.argmax()
        # outputs.append(z.data.numpy())
        print(f"{nn.functional.softmax(o, dim=0)} {z} : {y} =", t)
        rigits += int(t)
    len(x_test) and print(rigits / len(x_test))

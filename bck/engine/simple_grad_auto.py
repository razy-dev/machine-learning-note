import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim

from bck.engine.dataset import dataset, load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing


class NeuralNetwork(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1):
        super(NeuralNetwork, self).__init__()

        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.do = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o, _ = self.rnn(x)
        o = self.do(o[-1]) if self.num_layers > 1 else o[-1]
        z = self.lin(o)
        return z


costs = {}
accuracies = {}


def optimizer(ts, hs, nl, es, lr):
    cond = f"ts = {ts}, hs = {hs}, nl = {nl}, es = {es}, lr = {lr}"

    x_train, y_train, x_test, y_test, scaler = dataset(load(size=ts * 100), window_size=ts, output_size=1, normalizer='gradient')
    print(f"\nDataset for {cond}\n", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = NeuralNetwork(input_size=3, hidden_size=3 * hs, output_size=3, num_layers=nl)
    criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()  # couse fo is nn.LogSoftmax
    optimizer = optim.Adam(model.parameters(), lr)

    print(f"\nTraining ... {cond}")
    for e in range(es):
        for i, x in enumerate(x_train):
            z: Tensor = model(x)
            y: Tensor = y_train[i].view(-1)
            cost: Tensor = criterion(z, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if not costs or min(costs.keys()) > cost:
                costs[cost.item()] = {'cond': cond, 'e': e, 'i': i, 'z': z.data.numpy(), 'y': y.data.numpy(), 'cost': cost.item()}
        print(f"{cond} : {e} : {cost}")

    print(f"\nValidating ... {cond}")
    rigits = 0
    with torch.no_grad():
        for i, x in enumerate(x_test):
            o = model(x)
            z = np.eye(3)[o.data.numpy().argmax()]
            y = y_test.squeeze()[i].data.numpy()
            t = z.argmax() == y.argmax()
            rigits += int(t)
        accuracy = rigits / len(x_test) if len(x_test) else 0
        accuracies[accuracy] = cond
        print(f"{cond} : {accuracy}")


lr = 0.01
for ts in range(10, 100, 10):
    for hs in range(1, 5):
        for nl in range(1, 5):
            for es in range(100, 100 + 1, 100):
                optimizer(ts, hs, nl, es, lr)

# Export
import pandas as pd

c = pd.DataFrame(costs)
a = pd.DataFrame(accuracies)
with pd.ExcelWriter('output.xlsx') as writer:
    c.to_excel(writer, sheet_name='Costs')
    a.to_excel(writer, sheet_name='Accuracies')

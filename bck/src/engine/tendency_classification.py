import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch import optim

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing
warnings.filterwarnings("ignore")

debug = False

# Hyperparameter
time_steps = 3
features = 1
hidden_size = 1
output_size = 1

# Dataset
data_size = 100
train_size = round(data_size * 0.7)
data = np.random.random(data_size + 1 + time_steps) * 1000

scaler = MinMaxScaler()
grads = np.array([data[i + 1] - data[i] for i in range(len(data) - 1)])
grads_scaled = scaler.fit_transform(grads.reshape(-1, 1))


def _build_dataset(data, time_steps: int, output_size: int = 1):
    x, y = [], []
    for i in range(time_steps, len(data) - output_size):
        x.append(data[i - time_steps:i])
        y.append(data[i:i + output_size])
    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))


x_train, y_train = _build_dataset(grads_scaled[:train_size], time_steps, output_size)
debug and print(x_train.shape, x_train, y_train.shape, y_train)

x_test, y_test = _build_dataset(grads_scaled[train_size:], time_steps, output_size)
debug and print(x_test.shape, x_test, y_test.shape, y_test)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.RNN(features, output_size)
        self.fo = nn.Softmax()

    def forward(self, x):
        o, _ = self.layer(x)
        z = o[-1]  # self.fo(o[-1])
        return z


model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.01)

for e in range(100):
    for i, x in enumerate(x_train):
        z = model(x)
        cost = criterion(z, y_train[i])

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    print(z, y_train[i], cost)

if __name__ == "__main__":
    print('')
    pass

import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch import optim

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

time_steps = 3
features = 1
hidden_size = 1
output_size = 1

scaler = MinMaxScaler()
data_size = 10
data = yf.download('AMZN', start='2013-01-01', end='2021-12-31', progress=False)[['Adj Close']][:data_size * 2].values
data = scaler.fit_transform(data)


def _y(y0, y1):
    return 0 if y1 < y0 else 1


x_train, y_train, x_test, y_test = [], [], [], []
for i in range(time_steps, data_size):
    x_train.append(data[i - time_steps: i])
    y_train.append(_y(data[i - 1], data[i]))
    ti = i + data_size
    x_test.append(data[ti - time_steps: ti])
    y_test.append(_y(data[ti - 1], data[ti]))

x_train = torch.FloatTensor(np.array(x_train))
y_train = torch.FloatTensor(np.array(y_train)).unsqueeze(dim=1)
x_test = torch.FloatTensor(np.array(x_test))
y_test = torch.FloatTensor(np.array(y_test)).unsqueeze(dim=1)


# print(x_train, y_train, x_test, y_test)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.RNN(input_size, 1, num_layers=2),
        )
        self.ac = nn.Sigmoid()

    def forward(self, x):
        o, _ = self.layer(x)
        z = o[-1]
        return self.ac(z)


model = Model(features, hidden_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), 0.01)

for e in range(10):
    for i, x in enumerate(x_train):
        z = model(x)
        cost = criterion(z, y_train[i])

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(z, y_train[i], cost)

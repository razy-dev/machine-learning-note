import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch import optim

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing

scaler = MinMaxScaler()

time_steps = 3


def build_dataset(data, time_steps: int = 3):
    x, y = [], []
    for i in range(time_steps, len(data)):
        x.append(data[i - time_steps:i])
        y.append(data[i])
    return torch.FloatTensor(np.array(x)).unsqueeze(dim=2), torch.FloatTensor(np.array(y)).unsqueeze(dim=1)


x_train, y_train = build_dataset(data=np.linspace(1, 10, 10, dtype=float))
print(x_train, y_train)

x_test, y_test = build_dataset(data=np.linspace(11, 20, 10, dtype=float))
print(x_test, y_test)

model = nn.RNN(input_size=1, hidden_size=5, num_layers=1, nonlinearity='relu')
criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.01)
for e in range(20):
    for i, x in enumerate(x_train):
        o, _ = model(x)
        z = o[-1, -1:]
        cost = criterion(z, y_train[i])
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    print(z, y_train[i], cost)

with torch.no_grad():
    for i, x in enumerate(x_test):
        o, _ = model(x)
        z = o[-1, -1:]
        print(x, z, z.round(), y_test[i])

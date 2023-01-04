import torch
import torch.nn as nn
from torch import Tensor

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

time_steps = 3
features = 1
hidden_size = 1
output_size = 1

data = torch.FloatTensor([
    [[0], [0], [0], [0]],
    # [[0], [0], [0], [1]],
    [[0], [0], [1], [0]],
    # [[0], [0], [1], [1]],

    # [[0], [1], [0], [0]],
    [[0], [1], [0], [1]],
    # [[0], [1], [1], [0]],
    [[0], [1], [1], [1]],

    [[1], [0], [0], [0]],
    # [[1], [0], [0], [1]],
    [[1], [0], [1], [0]],
    # [[1], [0], [1], [1]],

    # [[1], [1], [0], [0]],
    [[1], [1], [0], [1]],
    # [[1], [1], [1], [0]],
    [[1], [1], [1], [1]],
])
x_train = data[:, :3]
y_train = data[:, 3:].squeeze(dim=2)
print(x_train.shape, y_train.shape)


class Model(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1):
        super(Model, self).__init__()

        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.do = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o, _ = self.rnn(x)
        o = self.do(o[-1]) if self.num_layers > 1 else o[-1]
        z = self.lin(o)
        return z


model = Model(input_size=features, hidden_size=hidden_size, output_size=output_size, num_layers=1)
z = model(x_train[0])
pass

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(f"\nTraining ... ")
for e in range(10):
    for i, x in enumerate(x_train):
        z: Tensor = model(x)
        y: Tensor = y_train[i].view(-1)
        cost: Tensor = criterion(z, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print(f"{e} : {cost}")

# with torch.no_grad():
#     for i, x in enumerate(x_train):
#         z = model(x)
#         print(x, z.round(), y_train[i])

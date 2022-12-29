import torch
import torch.nn as nn

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
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.RNN(1, 2)
        self.fo = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        o, _ = self.layer(x)
        z = self.fo(o[-1])
        return z


model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for e in range(100):
    for i, x in enumerate(x_train):
        z = model(x)
        cost = criterion(z, y_train[i])
        print(x, z, y_train[i], cost)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

with torch.no_grad():
    for i, x in enumerate(x_train):
        z = model(x)
        print(x, z.round(), y_train[i])

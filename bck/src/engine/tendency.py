import torch
import torch.nn as nn
from torch import Tensor

from engine.dataset import dataset, load
from engine.normalizer import GradNormalizer

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

time_steps = 5
features = 1
hidden_size = 3
output_size = 3

x_train, y_train, x_test, y_test, scaler = dataset(
    load(size=1000),
    window_size=time_steps,
    output_size=1,
    features=features,
    normalizer=GradNormalizer()
)
print("Size : ", len(x_train), len(x_test))


class Model(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=2):
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


model = Model(input_size=features, hidden_size=128, output_size=2, num_layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nTraining ... ")
model.train()
for e in range(100):
    for i, x in enumerate(x_train):
        optimizer.zero_grad()
        z: Tensor = model(x).unsqueeze(dim=0)
        y: Tensor = torch.LongTensor([int(y_train[i] > 0)])
        cost: Tensor = criterion(z, y)
        cost.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    print(f"{e} : {cost}")

rigits = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for i, x in enumerate(x_test):
        z = model(x)
        zi = z.argmax()
        y = torch.LongTensor([int(y_train[i] > 0)])
        t = zi == y
        rigits += int(t)
        accuracy = (rigits / len(x_test) if len(x_test) else 0) * 100
        print(x, z, zi, y_test[i], y, t)
print(f"{rigits}/{len(x_test)} = {accuracy}%")

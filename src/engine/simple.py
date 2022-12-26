import torch
import torch.nn as nn
from torch import Tensor, optim

from engine.dataset import dataset, _load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing

time_steps = 5
in_features = 1
out_features = 1

x_train, y_train, x_test, y_test = dataset(_load(size=100), window_size=time_steps, features=in_features, output_size=out_features)


# print(x_train, y_train, x_test, y_test)

class Model(nn.Module):

    def __init__(self, type: str, input_size, hidden_size):
        super(Model, self).__init__()
        self.layer = self.load_model(type, input_size, hidden_size)

    def forward(self, x):
        outputs, _status = self.layer(x)
        return _status[-1:, -1:]

    def load_model(self, name, input_size, hidden_size):
        return getattr(nn, name)(input_size, hidden_size)


learning_rate = 0.01


def run(type: str, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor):
    model = Model(type, in_features, time_steps)
    criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    print('=' * 80)
    print(f"{type} Model")

    # Training
    for e in range(10):
        for i, x in enumerate(x_train):
            y_pred = model(x)
            cost = criterion(y_pred, y_train[i])
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        print(f"{e:>5} : {y_pred.item()} {y_train[i]} {cost}")

    # Validate
    pass
    print('\n\n')


run('RNN', x_train, y_train, x_test, y_test)

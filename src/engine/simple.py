import torch
import torch.nn as nn
from torch import Tensor, optim

from engine.dataset import dataset, load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing

time_steps = 5
in_features = 1
out_features = 1

x_train, y_train, x_test, y_test, scaler = dataset(load(size=100), window_size=time_steps, features=in_features, output_size=out_features)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# print(x_train, y_train, x_test, y_test)

class NeuralNetwork(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features) => z =
    def __init__(self, type: str, input_size: int, hidden_size: int, batch_first: bool = False):
        super(NeuralNetwork, self).__init__()
        self.layer = self.load_model(type, input_size, hidden_size, batch_first=batch_first)
        # self.af = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        outputs, _status = self.layer(x)
        y = outputs[:, -1]
        z = y[:, -1].unsqueeze(dim=1)
        return z

    def load_model(self, name, input_size, hidden_size, batch_first):
        return getattr(nn, name)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )


z = NeuralNetwork('RNN', in_features, time_steps)(torch.Tensor())
print(z.shape, z)

learning_rate = 0.001


def run(type: str, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor):
    import matplotlib.pyplot as plt

    model = NeuralNetwork(type, in_features, time_steps)
    criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    print('=' * 80)
    print(f"{type} Model")

    # Training
    least = 0
    costs = []
    for e in range(100):
        # for i, x in enumerate(x_train):
        y_pred = model(x_train)
        cost: Tensor = criterion(y_pred, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        costs.append(cost.item())
        print(f"{e:>5} : {cost}")
    # plt.plot(costs)

    # Validate
    outputs = []
    with torch.no_grad():
        for i, x in enumerate(x_test):
            y_pred: Tensor = model(x.unsqueeze(dim=0))
            outputs.append(y_pred.squeeze().item())
            print(f"{y_pred.item()} : {y_test[i]}")
    print('\n\n')
    plt.plot(y_test.squeeze(), "green")
    plt.plot(outputs, "red")
    plt.show()


if __name__ == "__main__":
    # run('RNN', x_train, y_train, x_test, y_test)
    # run('LSTM', x_train, y_train, x_test, y_test)
    pass

import torch
import torch.nn as nn
from torch import Tensor, optim

from engine.dataset import dataset, load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

time_steps = 5
features = 1
hidden_size = 1
output_size = 1

x_train, y_train, x_test, y_test, scaler = dataset(
    load(size=200),
    window_size=time_steps,
    features=6,
    output_size=output_size,
    scale=True
)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


class NeuralNetwork(nn.Module):
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(self, type: str, input_size: int, hidden_size: int, batch_first: bool = False):
        super(NeuralNetwork, self).__init__()
        self.layer = self.load_model(type, input_size, hidden_size, batch_first=batch_first)
        # self.af = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        outputs, _status = self.layer(x)
        y = outputs[-1]
        z = y
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


def run(type: str, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor):
    import matplotlib.pyplot as plt

    learning_rate = 0.01
    model = NeuralNetwork(type, features, hidden_size)
    criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    print('=' * 80)
    print(f"{type} Model")

    # Training
    least = 0
    costs = []
    for e in range(100):
        for i, x in enumerate(x_train):
            # z = (hidden_size, )
            z = model(x)
            # y_pred = (output_size, features) = (1,1)
            y_pred = z[-1:]

            # y = (output_size, features)
            y = y_train[i][:, 0]
            cost = criterion(y_pred, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # costs.append(cost.item())
        print(f"{e:>5} : {y_pred.item()} {y.item()} {cost}")
        # for param in model.parameters():
        #     print(param.name, param)
    # plt.plot(costs)

    # Validate
    outputs = []
    with torch.no_grad():
        for i, x in enumerate(x_test):
            z = model(x)
            y = y_test.squeeze()[i]
            outputs.append(z.item())
            print(f"{y_pred.item()} : {y_test[i]}")
    print('\n\n')
    plt.plot(y_test.squeeze(), "green")
    plt.plot(outputs, "red")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run('RNN', x_train, y_train, x_test, y_test)
    run('LSTM', x_train, y_train, x_test, y_test)
    pass

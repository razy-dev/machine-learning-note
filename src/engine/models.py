from torch import nn


class RnnModel(nn.Module):
    num_layers: int
    nonlinearity: str = 'tanh'
    bias: bool = True
    dropout: nn.Dropout
    bidirectional: bool = False

    # x = (k, n) = (time_steps, input_features)
    # z = (k, m) = (time_steps, hidden_size)
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            dropout: float = 0.3,
            batch_first: bool = False
    ):
        super(RnnModel, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layer = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o, _ = self.layer(x)
        o = o[:, -1] if self.batch_first else o[-1]
        o = self.dropout(o) if self.num_layers > 1 else o
        z = self.fc(o)
        return z

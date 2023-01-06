import numpy as np
import torch
import yfinance as yf
from pandas import DataFrame
from torch import Tensor

from engine.normalizer import Normalizer, GradNormalizer

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing


def load(ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None) -> DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)[
            ['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]
        ]
        return df[:min(len(df), size)]
    except Exception as e:
        print(e)
        return DataFrame([])


def _build_dataset(data: DataFrame, window_size: int, output_size: int) -> tuple:
    x = []
    y = []
    for i in range(window_size, len(data) - output_size + 1):
        x.append(data[i - window_size:i])
        y.append(data[i:i + output_size])
    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))


def _scale(data: np.ndarray) -> tuple:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def _gradient(data: np.ndarray) -> tuple:
    grad = [np.array([0.0])]
    for i in range(len(data) - 1):
        r = ((data[i + 1] - data[i]) / data[i])
        grad.append(r)
    return grad, None


def regulate(data, midrange: float = 0.001) -> Tensor:
    if abs(data) <= midrange:
        return torch.LongTensor([1])
    elif data < 0:
        return torch.LongTensor([0])
    else:
        return torch.LongTensor([2])


def dataset(
        data: DataFrame,
        window_size: int,
        output_size: int,
        features: int = 1,
        split: int = 70,
        normalizer: Normalizer = None
) -> tuple:
    data = data.iloc[:, :features].values
    data, scaler = normalizer(data) if normalizer else (data, None)

    total = len(data)
    train_size = round(total * split / 100)
    return _build_dataset(data[:train_size], window_size, output_size) \
           + _build_dataset(data[train_size:], window_size, output_size) \
           + (scaler,)


# for test
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scaler = dataset(
        load(size=50),
        window_size=3,
        output_size=1,
        features=1,
        normalizer=GradNormalizer()
    )
    print(x_train.shape)
    print(x_train)
    print(y_train.shape)
    print(y_train)
    print(x_test.shape)
    print(y_test.shape)
    print(scaler)

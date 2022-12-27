import numpy as np
import torch
import yfinance as yf
from pandas import DataFrame

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing


def load(ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None) -> DataFrame:
    try:
        return yf.download(ticker, start=start, end=end, progress=False)[['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]][:size]
    except Exception as e:
        print(e)
        return DataFrame([])


def _build_dataset(data: DataFrame, window_size: int, output_size: int) -> tuple:
    x = []
    y = []
    for i in range(window_size, len(data) - output_size + 1):
        x.append(data[i - window_size:i])
        y.append(data[i:i + output_size])
    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y)).squeeze(dim=1)


def dataset(data: DataFrame, window_size: int, output_size: int, features: int = 1, split: int = 70) -> tuple:
    from sklearn.preprocessing import MinMaxScaler
    data = data.iloc[:, :features].values.round(2)
    total = len(data)
    train_size = round(total * split / 100)
    scaler = MinMaxScaler()
    ds = scaler.fit_transform(data)
    return *(_build_dataset(ds[:train_size], window_size, output_size) + _build_dataset(ds[train_size:], window_size, output_size)), scaler


# for test
if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler

    d = torch.rand(5, 2) * 10
    print(d)
    s = MinMaxScaler()
    ds = s.fit_transform(d)
    print(ds)
    di = s.inverse_transform(ds)
    print(di)
# x_train, y_train, x_test, y_test = dataset(_load(size=50), 5, 2)
# print(x_train, y_train, x_test, y_test)

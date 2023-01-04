from typing import Union

import numpy as np
import yfinance as yf
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from engine.conf import COLUMNS
from engine.normalizer import Normalizer, GradNormalizer


class StockDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        pass


class DatasetBuilder:
    data: DataFrame

    def __init__(self, ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None):
        self.data = self.load(ticker, start, end, size)

    def build(self, time_steps: int, features: Union[int, list, dict] = None, output_size: int = 1, normalizer: Normalizer = None, train_rate: float = 0.7):
        data = self.features(self.data, features, normalizer)

        total = len(data)
        train_size = round(total * train_rate)
        return self.build_dataset(data[:train_size], time_steps, output_size), self.build_dataset(data[train_size:], time_steps, output_size)

    @classmethod
    def build_dataset(cls, data: ndarray, time_steps: int, output_size: int):
        return time_steps * output_size

    @classmethod
    def features(cls, df: DataFrame, features: Union[int, str, list, dict], normalizer: Normalizer = None) -> ndarray:
        if not features or isinstance(features, int):
            return cls.normalize(np.array(df.iloc[:, :features].values), normalizer)
        if features or isinstance(features, str):
            return cls.normalize(np.array(df[features].values), normalizer)
        elif features and isinstance(features, list):
            return cls.normalize(np.array((df.iloc[:, features] if isinstance(features[0], int) else df[features]).values), normalizer)
        elif features and isinstance(features, dict):
            data = []
            for column, normalizer in features.items():
                data.append(cls.normalize(np.array(df[column].values), normalizer))
            return np.concatenate(data, axis=1)

    @classmethod
    def normalize(cls, data: ndarray, normalizer: Normalizer, **kwargs) -> ndarray:
        data = data[:, np.newaxis] if data.ndim == 1 else data
        return normalizer.normalize(data, **kwargs) if normalizer else data

    @classmethod
    def load(cls, ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None) -> DataFrame:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)[
                [COLUMNS.ADJ_CLOSE, COLUMNS.OPEN, COLUMNS.HIGH, COLUMNS.LOW, COLUMNS.CLOSE, COLUMNS.VOLUME]
            ]
            return df.iloc[:min(len(df), size) if size else None]
        except Exception as e:
            print(e)
            return DataFrame([])


# for test
if __name__ == "__main__":
    adj_normalizer = GradNormalizer()
    builder = DatasetBuilder(size=10)
    train_dataset, test_dataset = builder.build(time_steps=3, features=1, output_size=2, normalizer=adj_normalizer)

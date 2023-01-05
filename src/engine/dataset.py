from __future__ import annotations

from typing import Union, Callable, Final

import numpy as np
import yfinance as yf
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from engine.dataformats import DataFormat, DefaultFormat, TimestepFormat
from engine.normalizers import Normalizer, GradScaleNormalizer
from engine.transforms import GradTargetTransform, TensorTransform


class DataBuffer:
    data: DataFrame
    normalizer: Normalizer

    def __init__(self, ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None, normalizer: Normalizer = None):
        self.data = self.load(ticker, start, end, size)
        self.normalizer = normalizer

    def read(self, features: Union[int, list, dict] = 1, train_rate: float = 0.7, normalizer: Normalizer = None):
        data = self.features(self.data, features, normalizer or self.normalizer)
        train_size = round(len(data) * train_rate)
        return data[:train_size], data[train_size:]

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
                [cls.COLUMN_ADJ_CLOSE, cls.COLUMN_OPEN, cls.COLUMN_HIGH, cls.COLUMN_LOW, cls.COLUMN_CLOSE, cls.COLUMN_VOLUME]
            ]
            return df.iloc[:min(len(df), size) if size else None]
        except Exception as e:
            print(e)
            return DataFrame([])

    COLUMN_ADJ_CLOSE: Final = 'Adj Close'
    COLUMN_OPEN: Final = 'Open'
    COLUMN_HIGH: Final = 'High'
    COLUMN_LOW: Final = 'Low'
    COLUMN_CLOSE: Final = 'Close'
    COLUMN_VOLUME: Final = 'Volume'


class StockDataset(Dataset):
    input_data: list
    input_transform: Callable
    target_data: list
    target_transform: Callable

    def __init__(self, data: tuple, input_transform: Callable = TensorTransform(), target_transform: Callable = TensorTransform()):
        self.input_data, self.target_data = data
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index) -> T_co:
        try:
            return self.transform(self.input_data[index], self.input_transform), self.transform(self.target_data[index], self.target_transform)
        except Exception as e:
            print(e)
            return None, None

    def __len__(self):
        return len(self.input_data)

    @classmethod
    def transform(cls, data, transform: Callable):
        return transform(data) if transform else data


class DatasetBuilder:
    data: tuple
    format: Callable
    input_transform: Callable
    target_transform: Callable

    def __init__(
            self,
            data: tuple = None,
            format: Callable = None,
            input_transform: Callable = None,
            target_transform: Callable = None,
    ):
        self.data = data
        self.format = format
        self.input_transform = input_transform
        self.target_transform = target_transform

    def build(
            self,
            data: tuple = None,
            format: DataFormat = None,
            input_transform: Callable = None,
            target_transform: Callable = None
    ):
        data = data or self.data
        format = format or self.format or DefaultFormat()
        input_transform = input_transform or self.input_transform
        target_transform = target_transform or self.target_transform

        # TODO:
        # assert buffer is not None, ''
        # assert time_steps is not None, ''
        # assert buffer is not None, ''
        # assert buffer is not None, ''

        return StockDataset(format(data[0]), input_transform, target_transform), StockDataset(format(data[1]), input_transform, target_transform)


# for test
if __name__ == "__main__":
    adj_normalizer = GradScaleNormalizer()
    train_dataset, test_dataset = DatasetBuilder().build(
        data=DataBuffer(size=100).read(
            features=2,
            normalizer=adj_normalizer,
            train_rate=0.7,
        ),
        format=TimestepFormat(
            time_steps=5,
            output_size=3,
        ),
        input_transform=None,
        target_transform=GradTargetTransform()
    )
    print(train_dataset, train_dataset[0])
    print(test_dataset, test_dataset[0])

    # train_dataloader = DataLoader(train_dataset, batch_size=8)
    # test_dataloader = DataLoader(test_dataset)
    # print(train_dataloader, test_dataloader)
    #
    # print(next(train_dataloader))

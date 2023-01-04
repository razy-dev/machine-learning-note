import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def normalize(self, data: np.ndarray, **kwargs):
        raise NotImplementedError

    def inverse(self, data, **kwargs):
        return data


class GradNormalizer(Normalizer):

    def __init__(self, **kwargs):
        self.scaler = MinMaxScaler(**kwargs)

    def normalize(self, data: np.ndarray, **kwargs):
        return self.scaler.fit_transform(np.gradient(data, axis=0))


class ScaleNormalizer(Normalizer):
    scaler: MinMaxScaler

    def __init__(self, **kwargs):
        self.scaler = MinMaxScaler(**kwargs)

    def normalize(self, data: np.ndarray, **kwargs):
        return self.scaler.fit_transform(data)

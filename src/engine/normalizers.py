import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def normalize(self, data: np.ndarray, **kwargs):
        raise NotImplementedError

    def inverse(self, data, **kwargs):
        return data


class GradNormalizer(Normalizer):
    def normalize(self, data: np.ndarray, **kwargs):
        return np.gradient(data, axis=0)


class ScaleNormalizer(Normalizer):
    scaler: MinMaxScaler

    def __init__(self, **kwargs):
        self.scaler = MinMaxScaler(**kwargs)

    def normalize(self, data: np.ndarray, **kwargs):
        return self.scaler.fit_transform(data)

    def inverse(self, data, **kwargs):
        return self.scaler.inverse_transform(data)


class GradScaleNormalizer(ScaleNormalizer):
    def normalize(self, data: np.ndarray, **kwargs):
        return super().normalize(np.gradient(data, axis=0))

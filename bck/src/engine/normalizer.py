import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def __init__(self, **kwargs):
        self.scaler = None
        self.params = kwargs
        self._initialize()

    def _initialize(self):
        pass

    def __call__(self, data) -> tuple:
        return self.normalize(data, **self.params), self.scaler

    def normalize(self, data, **kwargs):
        raise NotImplementedError


class ScaleNormalizer(Normalizer):
    def _initialize(self):
        self.scaler = MinMaxScaler()

    def normalize(self, data, **kwargs):
        return self.scaler.fit_transform(data, **kwargs)


class GradNormalizer(Normalizer):
    def normalize(self, data, **kwargs):
        grad = [np.array([0.0])]
        for i in range(len(data) - 1):
            r = ((data[i + 1] - data[i]) / data[i])
            grad.append(r)
        return grad


class ScaleTendNormalizer(Normalizer):
    def normalize(self, data, **kwargs):
        pass


class CustomNormalizer(Normalizer):
    def normalize(self, data, **kwargs):
        return []

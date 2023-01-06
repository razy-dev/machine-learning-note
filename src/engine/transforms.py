import numpy as np
import torch

# Function
from engine.normalizers import Normalizer


def to_tensor(data, *args, **kwargs):
    return torch.FloatTensor(data if isinstance(data, np.ndarray) else np.array(data))


def to_long_tensor(data, *args, **kwargs):
    return torch.LongTensor(data if isinstance(data, np.ndarray) else np.array(data))


# Callables
class Transform:
    def __call__(self, data, *args, **kwargs):
        return self.trans(data, *args, **kwargs)

    def trans(self, data, *args, **kwargs):
        raise NotImplementedError


class TensorTransform(Transform):
    def trans(self, data, *args, **kwargs):
        return to_tensor(data, *args, **kwargs)


class TernaryIndexTransform(Transform):
    min: float
    max: float
    normalizer: Normalizer

    def __init__(self, base: float = 0., midrange: float = 0.01, normalizer: Normalizer = None):
        self.min = max(0, base - midrange)
        self.max = min(1, base + midrange)
        self.normalizer = normalizer

    # TODO: np fromiter, vectorize ...
    def trans(self, data, *args, **kwargs):
        data = self.normalizer.inverse(data) if self.normalizer else data
        targets = data[:, :1].squeeze()
        return to_long_tensor(
            [self.to_index(targets)] if targets.ndim == 0 else [self.to_index(t) for t in targets]
        )

    def to_index(self, v):
        return 0 if v < self.min else (2 if v > self.max else 1)

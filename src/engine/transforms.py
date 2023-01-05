import numpy as np
import torch


# Function
def to_tensor(data, *args, **kwargs):
    return torch.FloatTensor(data if isinstance(data, np.ndarray) else np.array(data))


# Callables
class Transform:
    def __call__(self, data, *args, **kwargs):
        return self.trans(data, *args, **kwargs)

    def trans(self, data, *args, **kwargs):
        raise NotImplementedError


class TensorTransform(Transform):
    def trans(self, data, *args, **kwargs):
        return to_tensor(data, *args, **kwargs)


class GradTargetTransform(Transform):
    def trans(self, data, *args, **kwargs):
        targets = data[:, :1]
        # for i, v in enumerate(targets):
        #     targets[i] = v >= 0.5
        return to_tensor(targets)

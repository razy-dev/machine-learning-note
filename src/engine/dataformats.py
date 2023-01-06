import numpy as np


class DataFormat:
    def __call__(self, data, *args, **kwargs) -> tuple:
        return self.format(data, *args, **kwargs)

    def format(self, data, *args, **kwargs) -> tuple:
        raise NotImplementedError


class DefaultFormat(DataFormat):
    def format(self, data, **kwargs) -> tuple:
        return data


class TimestepFormat(DataFormat):
    time_steps: int
    target_size: int

    def __init__(self, time_steps: int = 1, target_size: int = 1):
        self.time_steps = time_steps or 1
        self.target_size = target_size or 1

    # input = (len, time_steps, features), target = (len, output_size, features)
    def format(self, input_data, target_data=None, time_steps: int = None, target_size: int = None) -> tuple:
        target_data = input_data if target_data is None else target_data
        time_steps = time_steps or self.time_steps
        target_size = target_size or self.target_size
        input = []
        target = []
        for i in range(time_steps, len(input_data) - target_size + 1):
            input.append(input_data[i - time_steps:i])
            target.append(target_data[i:i + target_size])
        return np.array(input), np.array(target)

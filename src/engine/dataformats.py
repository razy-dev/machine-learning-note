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
    output_size: int

    def __init__(self, time_steps: int = 1, output_size: int = 1):
        self.time_steps = time_steps or 1
        self.output_size = output_size or 1

    # input = (len, time_steps, features), target = (len, output_size, features)
    def format(self, data, time_steps: int = None, output_size: int = None) -> tuple:
        time_steps = time_steps or self.time_steps
        output_size = output_size or self.output_size
        input = []
        target = []
        for i in range(time_steps, len(data) - output_size + 1):
            input.append(data[i - time_steps:i])
            target.append(data[i:i + output_size])
        return np.array(input), np.array(target)

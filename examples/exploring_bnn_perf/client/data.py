"""
Managing the inputs and labels for the function under test.
"""

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch

from torch import Tensor
from torch.utils.data import Dataset


def f(a: Tensor, b: Tensor) -> Tensor:
    # f(a, b) = \frac{1}{5}a^{2}-\frac{1}{10}b^{3}
    return (1 / 5 * a**2) - (1 / 10 * b**3)


class FutData(Dataset):
    def __init__(self, data_size):

        root = int(np.sqrt(data_size))
        self.data_size = root**2
        if self.data_size != data_size:
            print(f"Truncated datasize to be a valid square: {self.data_size}")

        self.a = torch.linspace(-6, 6, root)
        self.b = torch.linspace(-6, 6, root)

        self.inputs = torch.cartesian_prod(self.a, self.b)
        self.targets = f(*self.inputs.T).reshape(-1, 1)

        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.inputs = torch.tensor(
            self.input_scaler.fit_transform(self.inputs), dtype=torch.float32
        )
        self.targets = torch.tensor(
            self.target_scaler.fit_transform(self.targets), dtype=torch.float32
        )

        self.input_min = self.inputs_std.min().item()
        self.input_max = self.inputs_std.max().item()

    @property
    def inputs_std(self):
        return self.inputs

    @property
    def inputs_nonstd(self):
        return self.input_scaler.inverse_transform(self.inputs)

    @property
    def targets_std(self):
        return self.targets

    @property
    def targets_nonstd(self):
        return self.target_scaler.inverse_transform(self.targets)

    def transform_request(self, x: Tensor) -> Tensor:
        return self.input_min + x * (self.input_max - self.input_min)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        "Returns (inputs, targets)"
        return (
            self.inputs[idx],
            self.targets[idx],
        )

    def __iter__(self):
        # Used for unpacking. e.g. x, y = futData
        yield self.inputs
        yield self.targets
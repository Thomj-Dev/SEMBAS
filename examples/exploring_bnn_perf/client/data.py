"""
Managing the inputs and labels for the function under test.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from torch import Tensor
from torch.utils.data import Dataset


def simple_2d_regr(a: Tensor, b: Tensor) -> Tensor:
    # f(a, b) = \frac{1}{5}a^{2}-\frac{1}{10}b^{3}
    return (1 / 5 * a**2) - (1 / 10 * b**3)


class FutData(Dataset):
    def __init__(self, inputs: Tensor, targets: Tensor):
        self.inputs = inputs
        self.targets = targets

        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.inputs = torch.tensor(
            self.input_scaler.fit_transform(self.inputs), dtype=torch.float32
        )
        self.targets = torch.tensor(
            self.target_scaler.fit_transform(self.targets), dtype=torch.float32
        )

        self.data_size = self.inputs.shape[0]

        self.input_min = self.inputs_std.min().item()
        self.input_max = self.inputs_std.max().item()

    @property
    def inputs_std(self):
        "Standardized inputs"
        return self.inputs

    @property
    def inputs_nonstd(self):
        "Non-standardized inputs"
        return self.input_scaler.inverse_transform(self.inputs)

    @property
    def targets_std(self):
        "Standardized labels"
        return self.targets

    @property
    def targets_nonstd(self):
        "Non-standardized labels"
        return self.target_scaler.inverse_transform(self.targets)

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

    def transform_request(self, x: Tensor) -> Tensor:
        "Transforms request from SEMBAS to the input domain of a model for @f"
        return self.input_min + x * (self.input_max - self.input_min)

    def inverse_transform_request(self, x: Tensor) -> Tensor:
        "Transforms from input domain to SEMBAS request domain (i.e. normalized domain)"
        return (x - self.input_min) / (self.input_max - self.input_min)


class Simple2DRegrData(FutData):
    """
    Manages the training data for the simple function @f. Includes properties for
    standardized and unstandardized input and output data. Used in conjunction with
    pytorch's DataLoader to manage creating training batches.
    """

    def __init__(self, data_size):
        root = int(np.sqrt(data_size))
        self.data_size = root**2
        if self.data_size != data_size:
            print(f"Truncated datasize to be a valid square: {self.data_size}")

        self.a = torch.linspace(-6, 6, root)
        self.b = torch.linspace(-6, 6, root)

        super().__init__(
            torch.cartesian_prod(self.a, self.b),
            simple_2d_regr(*self.inputs.T).reshape(-1, 1),
        )


class HighDimensionalRegrData(FutData):

    def __init__(self, path: str = ".data/california_housing.csv"):
        from sklearn.datasets import fetch_california_housing

        self.data: pd.DateOffset = fetch_california_housing(as_frame=True).frame

        super().__init__(
            self.data[
                [
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude",
                ]
            ],
            self.data[["MedHouseVal"]],
        )

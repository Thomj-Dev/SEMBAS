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


class RetailRegressionData(FutData):
    """
    Requires Online Retail II dataset to be in the specified folder.
    This data is for predicting volume of sales from historical data.
    """

    def __init__(self, data_path: str, stock_code="85048"):
        """
        @data_path : The path to the Kaggle data, must be downloaded from
            https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
        @stock_code : The ID of the product to model.
        """
        try:
            self.raw_data = pd.read_csv(data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find file as {data_path}. Be sure to download data from "
                "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci"
            )
        self.raw_data = self.raw_data.groupby(
            ["date", "StockCode"], as_index=False
        ).agg(
            volume=("Quantity", "sum"),
            price=("Price", "mean"),
        )
        # Cosine encoding ensures weekends are similar in value.
        # 1 - cos means we are measuring distance from weekend, makes it human readable
        self.raw_data["weekday"] = 1 - np.cos(
            pd.to_datetime(self.raw_data["date"]).dt.dayofweek / 6 * np.pi
        )
        # d/dx (1-cos) = sin, velocity to/from weekend. Sign is + when going away from
        # weekend and towards wednesday, and - when going away from wednesday to weekend.
        self.raw_data["derivative"] = np.sign(
            np.sin(pd.to_datetime(self.raw_data["date"]).dt.dayofweek / 6 * np.pi)
        )
        self.raw_data["day"] = pd.to_datetime(self.raw_data["date"]).dt.day
        self.raw_data["month"] = pd.to_datetime(self.raw_data["date"]).dt.month
        self.raw_data["year"] = pd.to_datetime(self.raw_data["date"]).dt.year

        super().__init__(
            torch.tensor(
                self.raw_data[["day", "month", "year", "derivative", "price"]]
            ).numpy(),
            torch.tensor(self.raw_data[["volume"]]).numpy(),
        )

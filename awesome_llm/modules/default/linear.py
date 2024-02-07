
import numpy as np
import torch

import torch.nn as nn


class Linear(nn.Module):
    """
    A simple linear layer.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the Linear layer.

        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample.
        """
        super().__init__()
        self.weight = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the Linear layer.

        :param x: Input tensor.
        :return: Output after applying linear transformation.
        """
        return x @ self.weight.t() + self.bias

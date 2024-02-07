
import numpy as np
import torch

import torch.nn as nn

class Embedding(nn.Module):
    """
    A simple embedding layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes the Embedding layer.

        :param num_embeddings: Size of the dictionary of embeddings.
        :param embedding_dim: The size of each embedding vector.
        """
        super().__init__()
        self.weight = torch.randn(num_embeddings, embedding_dim, requires_grad=True)        

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the Embedding layer.

        :param indices: Indices to be embedded.
        :return: Embedded indices.
        """
        return self.weight[indices]
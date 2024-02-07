
import numpy as np
import torch
from awesome_llm.modules.base import SimpleModule


class Embedding(SimpleModule):
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
        self.add_parameter('weight', self.weight)
        

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the Embedding layer.

        :param indices: Indices to be embedded.
        :return: Embedded indices.
        """
        return self.weight[indices]
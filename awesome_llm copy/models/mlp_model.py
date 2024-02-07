import logging
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel

g = torch.Generator().manual_seed(2147483647)


class MLPModel(BaseModel):
    """
    Multi Layer Perceptron model for name generation.
    """

    def __init__(self, 
                 num_classes: int, 
                 embedding_size: int, 
                 block_size: int):
        """
        Initializes the neural network model.

        :param num_classes: Number of classes (unique characters in the dataset).
        :param embedding_size: Size of the embedding layer.
        :param block_size: Size of the 'context window'.
        """
        super().__init__()

        # Size of the 'context window'
        self.block_size = block_size

        # Embeddings layer
        # Using nn.Embedding:
        # Initialize an embedding layer with random values.
        self.embedding = nn.Embedding(num_classes, embedding_size)

        # Using torch.rand:
        # Directly initialize an embedding matrix with random values.
        # Key difference is that nn.Embedding is a PyTorch module designed for embedding layers in neural networks and allows for the learning of embeddings during training, whereas torch.rand simply initializes the embedding matrix with random values without any learning capability
        # self.embedding = torch.rand((num_classes, embedding_size))

        # Define your hidden layers here
        self.hidden_layers = nn.Sequential(
            nn.Linear(embedding_size * block_size, 128),  # Example hidden layer with 128 units
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        :param x: Input tensor (one-hot encoded).
        :return: Output tensor.
        """
        # Embedding layer
        x = self.embedding(x)

        # Reshape the tensor for processing through hidden layers
        x = x.view(x.size(0), -1)  # Flatten the input

        # Pass through hidden layers
        x = self.hidden_layers(x)

        return x

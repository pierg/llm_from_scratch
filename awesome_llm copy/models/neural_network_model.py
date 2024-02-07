from models.base_model import BaseModel
import torch
import torch.nn.functional as F
import logging

g = torch.Generator().manual_seed(2147483647)

class NeuralNetworkModel(BaseModel):
    """
    One-layer neural network model for name generation.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the neural network model.

        :param num_classes: Number of classes (unique characters in the dataset).
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initializes the weights of the neural network.
        self.W = torch.randn((num_classes, num_classes), requires_grad=True, generator=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        :param x: Input tensor (one-hot encoded).
        :return: Output tensor.
        """
        logits = x @ self.W  # Predict log-counts
        counts = logits.exp()  # Counts, equivalent to N
        sum_counts = counts.sum(dim=1, keepdim=True)  # Sum counts along dimension 1
        probs = counts / sum_counts  # Softmax function for probabilities

        self.logger.debug("logits shape: %s", logits.shape)
        self.logger.debug("counts shape: %s", counts.shape)
        self.logger.debug("sum_counts shape: %s", sum_counts.shape)
        self.logger.debug("probs shape: %s", probs.shape)
        
        return probs
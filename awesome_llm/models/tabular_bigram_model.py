import torch
import torch.nn.functional as F
from typing import Tuple
from models.base_model import BaseModel

class TabularBigramModel(BaseModel):
    """
    Tabular bigram model for name generation.

    This model uses a precomputed probability matrix to generate the probability distribution of the next character.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        super().__init__()

        # Initializes the probability table
        self.P = None

    def set_probabilities(self, probabilities: torch.Tensor) -> None:
        """
        Sets the bigram probabilities matrix for the model.

        :param probabilities: A matrix of smoothed bigram probabilities.
        """
        self.P = probabilities

    def forward(self, input_index: int) -> torch.Tensor:
        """
        Performs the forward pass, returning the probability distribution of the next character.

        :param input_index: The index of the current character.
        :return: Probability distribution of the next character.
        """
        return self.P[input_index]

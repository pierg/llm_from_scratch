import torch
from data.base_data_loader import BaseDataLoader


class BaseModel(torch.nn.Module):
    """
    Abstract base class for machine learning models.

    This class serves as a blueprint for various machine learning models.
    It extends torch.nn.Module and defines common interfaces and methods.
    """

    def __init__(self):
        """
        Initializes the BaseModel instance.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        This method needs to be overridden by subclasses.

        :param x: Input tensor.
        :return: Output tensor.
        """
        raise NotImplementedError("The forward method must be overridden by the subclass.")

    def save(self, path: str) -> None:
        """
        Saves the model weights to a file.

        :param path: File path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Loads the model weights from a file.

        :param path: File path to load the model.
        """
        self.load_state_dict(torch.load(path))

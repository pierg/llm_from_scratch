from pathlib import Path
import torch

class BaseDataLoader:
    """
    BaseDataLoader class for handling data loading and preprocessing.

    This class is designed to read data from files, preprocess it, and provide
    methods to access the processed data in a format suitable for training and
    prediction tasks in machine learning models.
    """

    def __init__(self, filename: Path, n_elements: int = None) -> None:
        """
        Initializes the BaseDataLoader with a dataset.

        :param filename: A path to the file containing data.
        :param n_elements: Number of elements to read from the file. If None, reads all entries.
        """
        self.data = self.read_data(filename, n_elements)

    def read_data(self, filename: Path, n_elements: int = None) -> list:
        """
        Reads data from a file.

        :param filename: A path to the file.
        :param n_elements: Number of elements to read from the file. If None, reads all entries.
        :return: A list of data items.
        """
        with open(filename, "r") as file:
            data = file.read().splitlines()
        if n_elements is not None:
            data = data[:n_elements]
        return data

    def preprocess_data(self) -> None:
        """
        Preprocesses the loaded data.

        This method should be implemented to perform tasks like normalization,
        tokenization, vectorization, etc., as per the requirements of the model.
        """
        raise NotImplementedError("This method should be overridden in a subclass.")

    def get_data(self) -> torch.Tensor:
        """
        Provides preprocessed data in a format ready for model consumption.

        :return: A torch.Tensor representing the preprocessed data.
        """
        raise NotImplementedError("This method should be overridden in a subclass.")

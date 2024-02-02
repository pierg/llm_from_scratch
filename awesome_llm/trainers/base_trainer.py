from data.base_data_loader import BaseDataLoader
from models.base_model import BaseModel

class BaseTrainer:
    """
    Abstract base class for training machine learning models.

    This class defines a generic interface for training routines, including methods
    for training loops, evaluation, and logging.
    """

    def __init__(self, model: BaseModel, data_loader: BaseDataLoader) -> None:
        """
        Initializes the BaseTrainer with a model and a data loader.

        :param model: An instance of a BaseModel.
        :param data_loader: An instance of a BaseDataLoader.
        """
        self.model = model
        self.data_loader = data_loader

    def train(self, epochs: int) -> None:
        """
        Trains the model.

        This method needs to be overridden by subclasses.

        :param epochs: Number of epochs to train the model.
        """
        raise NotImplementedError("The train method must be overridden by the subclass.")

    def evaluate(self) -> None:
        """
        Evaluates the model's performance.

        This method needs to be overridden by subclasses.
        """
        raise NotImplementedError("The evaluate method must be overridden by the subclass.")

    def log(self, message: str) -> None:
        """
        Logs a message, potentially to a file, console, or a monitoring tool.

        This method can be overridden by subclasses for customized logging.

        :param message: The message to be logged.
        """
        print(message)

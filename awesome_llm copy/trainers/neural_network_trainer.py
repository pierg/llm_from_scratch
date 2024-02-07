from trainers.base_trainer import BaseTrainer
from models.neural_network_model import NeuralNetworkModel
from data.names_data_loader import NamesDataLoader
import torch
import torch.nn.functional as F


class NeuralNetworkTrainer(BaseTrainer):
    """
    BaseTrainer for the one-layer neural network model.
    """

    def __init__(self, model: NeuralNetworkModel, data_loader: NamesDataLoader):
        """
        Initializes the neural network trainer.

        :param model: An instance of NeuralNetworkModel.
        :param data_loader: An instance of NamesDataLoader.
        """
        super().__init__(model, data_loader)
        self.xs, self.ys = self.data_loader.create_dataset()

    def train(self, learning_rate=0.01, epochs=1, reg_lambda=0.01):
        """
        Trains the neural network model.

        :param learning_rate: Learning rate for gradient descent.
        :param epochs: Number of training epochs.
        :param reg_lambda: Regularization parameter.
        """
        self.model.initialize_weights()

        for epoch in range(epochs):
            xenc = F.one_hot(self.xs, num_classes=len(self.data_loader.stoi)).float()
            probs = self.model(xenc)

            # Compute loss with regularization
            loss = -torch.log(probs[torch.arange(len(self.xs)), self.ys]).mean()
            loss += reg_lambda * (self.model.W ** 2).mean()  # Regularization

            # Backward pass and update weights
            self.model.W.grad = None
            loss.backward()
            with torch.no_grad():
                self.model.W -= learning_rate * self.model.W.grad

            print(f'Epoch {epoch}, Loss: {loss.item()}')

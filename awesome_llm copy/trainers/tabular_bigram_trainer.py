from trainers.base_trainer import BaseTrainer
from models.tabular_bigram_model import TabularBigramModel
from data.names_data_loader import NamesDataLoader
import torch


class TabularBigramTrainer(BaseTrainer):
    """
    Trainer for the TabularBigramModel.
    """

    def __init__(self, model: TabularBigramModel, data_loader: NamesDataLoader):
        """
        Initializes the TabularBigramTrainer.

        :param model: An instance of TabularBigramModel.
        :param data_loader: An instance of NamesDataLoader.
        """
        super().__init__(model, data_loader)
        self.compute_probabilities(data_loader)
        

    def compute_probabilities(self, data_loader: NamesDataLoader) -> None:
        """
        Computes and sets the bigram counts and probabilities matrices in the model.

        :param data_loader: An instance of NamesDataLoader containing the data.
        """
        counts_matrix = self._calculate_bigram_counts_matrix(data_loader)
        probabilities_matrix = self._calculate_bigram_probabilities_matrix(counts_matrix)
        self.model.set_probabilities(probabilities_matrix)

    def _calculate_bigram_counts_matrix(self, data_loader: NamesDataLoader) -> torch.Tensor:
        N = torch.zeros((len(data_loader.stoi), len(data_loader.stoi)), dtype=torch.int32)
        for word in data_loader.data:
            w = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(w, w[1:]):
                ix1, ix2 = data_loader.stoi.get(ch1, 0), data_loader.stoi.get(ch2, 0)
                N[ix1, ix2] += 1
        return N

    def _calculate_bigram_probabilities_matrix(self, count_matrix: torch.Tensor) -> torch.Tensor:
        smoothed_count_matrix = count_matrix + 1
        P = smoothed_count_matrix.float()
        P = P / P.sum(axis=1, keepdim=True)
        return P

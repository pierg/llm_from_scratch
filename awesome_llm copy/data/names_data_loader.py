from typing import List, Tuple, Dict
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data.base_data_loader import BaseDataLoader
import logging
import torch.nn.functional as F
from pathlib import Path
from data.base_data_loader import BaseDataLoader 

class NamesDataLoader(BaseDataLoader):
    """
    BaseDataLoader subclass for handling names data.

    This class is tailored for reading and preprocessing a list of names, typically
    stored one name per line in a text file.
    """

    def __init__(self, filename: Path, n_elements: int = None) -> None:
        """
        Initializes the NamesDataLoader with a dataset of names.

        :param filename: A path to the file containing names.
        :param n_elements: Number of elements to read from the file. If None, reads all entries.
        """
        super().__init__(filename, n_elements)
        self.stoi, self.itos = self.create_mappings()
        
        self.logger = logging.getLogger(__name__)

    def create_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Creates mappings from characters to indices and vice versa.

        :return: A tuple of two dictionaries (stoi, itos).
        """
        all_chars = "".join(self.data)
        unique_chars = sorted(set(all_chars))
        
        stoi = {char: idx for idx, char in enumerate(unique_chars, start=1)}
        stoi["."] = 0
        sorted_stoi = dict(sorted(stoi.items(), key=lambda x: x[1]))
        
        itos = {idx: char for char, idx in sorted_stoi.items()}
        
        return sorted_stoi, itos

    def char_to_one_hot(self, char: str) -> torch.Tensor:
        """
        Convert a character to its one-hot encoding tensor representation.

        :param char: The character to be converted.
        :return: A one-hot encoding tensor.
        """
        if char not in self.stoi:
            raise ValueError(f"Character '{char}' is not in the mapping.")
        idx = self.stoi[char]
        one_hot = F.one_hot(torch.tensor(idx), num_classes=len(self.stoi)).float()
        return one_hot
    

    def create_dataset(self, block_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a dataset of bigrams from the names.

        :param block_size: Context length, i.e., how many characters to use for prediction.
        :return: A tuple of tensors (xs, ys) representing input and target indices.
        """
        self.logger.info("Dataset creation...")

        xs = []
        ys = []
        
        for word in self.data:
            chs = list(word) + ['.']
            context = [self.stoi['.']] * block_size  # Pad the context with '.'

            for ch in chs:
                if ch in self.stoi:
                    ix = self.stoi[ch]
                    xs.append(context.copy())  # Append a copy of the context
                    ys.append(ix)
                    
                    # Log only the first 10 iterations for debugging
                    if len(xs) <= 10:
                        self.logger.debug('%s ---> %s', ''.join(self.itos[i] for i in context), self.itos[ix])
                    
                    context = context[1:] + [ix]  # Update context by shifting and adding
      

        # Convert lists to tensors
        xs_tensor = torch.tensor(xs, dtype=torch.int64)
        ys_tensor = torch.tensor(ys, dtype=torch.int64)

        # Debug logging messages for tensor shapes and dtypes
        self.logger.debug("Shape of xs_tensor: %s", xs_tensor.shape)
        self.logger.debug("Shape of ys_tensor: %s", ys_tensor.shape)
        self.logger.debug("dtype of xs_tensor: %s", xs_tensor.dtype)
        self.logger.debug("dtype of ys_tensor: %s", ys_tensor.dtype)
                
        self.logger.info("Dataset creation completed")

        return xs_tensor, ys_tensor
    
    
    def create_dataset2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a dataset of bigrams from the names.

        :return: A tuple of tensors (xs, ys) representing input and target indices.
        """
        self.logger.info("Creating dataset...")
        
        xs = []
        ys = []
        for word in self.data:
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                if ch1 in self.stoi and ch2 in self.stoi:
                    ix1 = self.stoi[ch1]
                    ix2 = self.stoi[ch2]
                    xs.append(ix1)
                    ys.append(ix2)
        
        # Debug logging messages
        self.logger.debug("Number of xs: %s", len(xs))
        self.logger.debug("Number of ys: %s", len(ys))
        
        # Convert lists to tensors
        xs_tensor = torch.tensor(xs, dtype=torch.int64)
        ys_tensor = torch.tensor(ys, dtype=torch.int64)

        # Debug logging messages for tensor shapes
        self.logger.debug("Shape of xs_tensor: %s", xs_tensor.shape)
        self.logger.debug("Shape of ys_tensor: %s", ys_tensor.shape)

        self.logger.info("Dataset creation complete.")
        
        return xs_tensor, ys_tensor
    

    def save_bigram_plot(self, bigrams: Dict[Tuple[str, str], int], filename: str) -> None:
        """
        Saves a bar plot of the bigram frequencies.

        :param bigrams: A dictionary of bigram counts.
        :param filename: The filename for the saved plot.
        """
        plt.figure(figsize=(20, 3))
        plt.bar([a + b for a, b in bigrams.keys()][:50], list(bigrams.values())[:50])
        plt.xticks(rotation=45)
        plt.title("Bigram Frequencies")
        plt.tight_layout()
        plt.savefig(filename)

    def save_heatmap(self, data: torch.Tensor, filename: str) -> None:
        """
        Saves a heatmap of the bigram probabilities.

        :param data: A matrix of bigram probabilities.
        :param filename: The filename for the saved heatmap.
        """
        plt.figure(figsize=(20, 20))
        sns.heatmap(data, xticklabels=list(self.stoi.keys()), yticklabels=list(self.stoi.keys()), annot=True, fmt="g")
        plt.title("Bigram Probabilities Heatmap")
        plt.xlabel("Second Letter")
        plt.ylabel("First Letter")
        plt.tight_layout()
        plt.savefig(filename)
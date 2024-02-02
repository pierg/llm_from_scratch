import torch
from pathlib import Path
from typing import Tuple, List
from educational.utils import create_mappings, read_words
import random

def build_dataset(words: List[str], stoi: dict, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a dataset from the given words.

    Args:
    words (List[str]): List of words to create the dataset from.
    stoi (dict): A mapping from characters to their integer representation.
    block_size (int): The size of the context block for prediction.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Tensors representing the input and target datasets.
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context[:])
            Y.append(ix)
            context = context[1:] + [ix]  # Crop and append

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    print(f"Dataset Shape - X: {X.shape}, Y: {Y.shape}")
    print(f"Dataset Dtype - X: {X.dtype}, Y: {Y.dtype}")
    return X, Y

def create_dataset_with_context(filepath: Path, block_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates training, development, and test datasets from a file.

    Args:
    filepath (Path): Path to the file containing words.
    block_size (int): The size of the context block for prediction.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
    Tensors representing the training, development, and test datasets (Xtr, Ytr, Xdev, Ydev, Xte, Yte).
    """
    words = read_words(filepath)
    stoi, _ = create_mappings(words)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)     # 80%
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size) # 10%
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)     # 10%

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

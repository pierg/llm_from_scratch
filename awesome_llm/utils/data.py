import torch


def split_data(data: torch.Tensor, train_pct: float = 0.7, dev_pct: float = 0.15) -> tuple:
    """
    Splits the data tensor into training, development, and testing sets.
    'train_pct' and 'dev_pct' are the proportions for training and development sets.
    """
    total_size = len(data)
    train_size = int(total_size * train_pct)
    dev_size = int(total_size * dev_pct)

    train_data = data[:train_size]
    dev_data = data[train_size:train_size+dev_size]
    test_data = data[train_size+dev_size:]

    return train_data, dev_data, test_data

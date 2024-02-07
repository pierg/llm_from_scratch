import torch
from pathlib import Path
from models.mlp_model import MLPModel
from trainers.mpl_trainer import MLPTrainer

from data.names_data_loader import NamesDataLoader

import os
import logging

# Set the path of the log file
log_file_path = 'my_log.log'

# Check if the log file exists
if os.path.exists(log_file_path):
    # Delete the log file
    os.remove(log_file_path)


logging.basicConfig(filename=log_file_path, level=logging.DEBUG)


g = torch.Generator().manual_seed(2147483647)


# Initialize data loader with a sample dataset
file_folder = Path(__file__).parent
data_loader = NamesDataLoader(file_folder.parent / "data" / "names.txt", n_elements=5)

# Initialize tabular model and trainer
neural_network_model = MLPModel(num_classes=27, block_size=3)
neural_network_trainer = MLPTrainer(neural_network_model, data_loader)

# Get model and data_loader from the trainer
model, data_loader = neural_network_trainer.model, neural_network_trainer.data_loader



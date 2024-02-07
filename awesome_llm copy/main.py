import torch
from pathlib import Path
from awesome_llm.modules.mlp_model import MLPModel

from data.names_data_loader import NamesDataLoader

from models.tabular_bigram_model import TabularBigramModel
from models.neural_network_model import NeuralNetworkModel
from models.mlp_model import MLPModel
from trainers.tabular_bigram_trainer import TabularBigramTrainer
from trainers.neural_network_trainer import NeuralNetworkTrainer
from trainers.mpl_trainer import MLPTrainer
from utils.helpers import illustrate_forward_process, illustrate_backward_process, generate_name
import os, logging

# Set the path of the log file
log_file_path = 'my_log.log'

# Check if the log file exists
if os.path.exists(log_file_path):
    # Delete the log file
    os.remove(log_file_path)


logging.basicConfig(filename=log_file_path, level=logging.DEBUG)


g = torch.Generator().manual_seed(2147483647)


def main_tabular():

    # Initialize data loader with a sample dataset
    file_folder = Path(__file__).parent
    data_loader = NamesDataLoader(file_folder.parent / "data" / "names.txt")

    # Initialize tabular model and trainer
    tabular_model = TabularBigramModel()
    tabular_trainer = TabularBigramTrainer(tabular_model, data_loader)

    # Get model and data_loader from the trainer
    model, data_loader = tabular_trainer.model, tabular_trainer.data_loader

    # Query the model for a single character prediction
    input_char = "."
    probs = model.forward(data_loader.stoi[input_char])
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    output_char = data_loader.itos[ix]
    print(f"Given {input_char} as input, the model has produced {output_char}")

    # Generate names
    num_names_to_generate = 5  # Set the desired number of names to generate
    generated_names = [generate_name(model, data_loader) for _ in range(num_names_to_generate)]

    # Print the generated names
    print("Generated names:")
    for i, name in enumerate(generated_names):
        print(f"Name {i + 1}: {name}")


def main_neural_network():

    # Initialize data loader with a sample dataset
    file_folder = Path(__file__).parent
    data_loader = NamesDataLoader(file_folder.parent / "data" / "names.txt")

    # Initialize tabular model and trainer
    neural_network_model = NeuralNetworkModel(num_classes=27)
    neural_network_trainer = NeuralNetworkTrainer(neural_network_model, data_loader)

    # Get model and data_loader from the trainer
    model, data_loader = neural_network_trainer.model, neural_network_trainer.data_loader

    # Query the model for a single character prediction
    input_char = "."
    probs = model.forward(data_loader.char_to_one_hot(input_char))
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    output_char = data_loader.itos[ix]
    print(f"Given {input_char} as input, the model has produced {output_char}")

    # Generate names
    num_names_to_generate = 5  # Set the desired number of names to generate
    generated_names = [generate_name(model, data_loader) for _ in range(num_names_to_generate)]

    # Print the generated names
    print("Generated names:")
    for i, name in enumerate(generated_names):
        print(f"Name {i + 1}: {name}")



def main_mlp():

    # Initialize data loader with a sample dataset
    file_folder = Path(__file__).parent
    data_loader = NamesDataLoader(file_folder.parent / "data" / "names.txt")

    # Initialize tabular model and trainer
    neural_network_model = MLPModel(num_classes=27, block_size=3)
    neural_network_trainer = MLPTrainer(neural_network_model, data_loader)

    # Get model and data_loader from the trainer
    model, data_loader = neural_network_trainer.model, neural_network_trainer.data_loader


if __name__ == "__main__":
    # main_tabular()
    # main_neural_network()
    main_mlp()


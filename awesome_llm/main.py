from pathlib import Path
import torch
from awesome_llm.optimizers.adamw import AdamW
from awesome_llm.trainers.trainer import Trainer
from awesome_llm.utils.torch import save_model_info
from data.text_data import TextDataLoader, CharacterTokenizer, TextBatchGenerator
from utils.data import split_data
from torch import nn
from awesome_llm.modules.model_v1 import GPT_v1 
from awesome_llm.modules.model_v2 import GPT_v2
from awesome_llm.modules.model_v3 import GPT_v3

# Configuration dictionary for easy hyperparameters adjustments and other settings
config = {
    "file_path": Path(__file__).parent / 'tiny-shakespeare.txt',
    "train_test_split": 0.9,
    "batch_size": 32,
    "block_size": 8,
    "max_iters": 3000,
    "eval_interval": 300,
    "learning_rate": 1e-2,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}

def process_data(config:dict):
    # Load and Tokenize Data
    data_loader = TextDataLoader()
    raw_text = data_loader.load_data(config["file_path"])
    tokenizer = CharacterTokenizer(raw_text)
    tokenized_text = tokenizer.tokenize(raw_text)

    # Split Data into Training, Validation, and Testing Sets
    train_data, val_data, test_data = split_data(tokenized_text, train_pct=config["train_test_split"])

    # Initialize Batch Generator
    data_splits = {'train': train_data, 'val': val_data, 'test': test_data}
    batch_generator = TextBatchGenerator(data_splits, block_size=config["block_size"])

    vocab_size = tokenizer.vocab_size()
    return vocab_size, tokenizer, batch_generator


vocab_size, tokenizer, batch_generator = process_data(config["file_path"])
dummy_input=torch.zeros((1, 1), dtype=torch.long, device=config["device"])
save_folder = folder=Path(__file__).parent / "info"


def training_loop(model: nn.Module, hyperparameters={}):
    # Update config with any provided hyperparameters
    config.update(hyperparameters)

    try:

        print("Generate Text from Undtrained Model...")
        # Generate Text from Undtrained Model
        context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])  # Starting context for text generation
        generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
        print(tokenizer.decode(generated_tokens))
        
        print("Training the model...")
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        trainer = Trainer(data=batch_generator, model=model, optimizer=optimizer, loss_fn=torch.nn.functional.cross_entropy)

        # Step 5: Train the Model
        trainer.train(config["batch_size"], config["max_iters"], config["eval_interval"])

        # Step 6: Generate Text from the Trained Model
        context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])  # Starting context for text generation
        generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
        print(tokenizer.decode(generated_tokens))

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":


    model_1 = GPT_v1(vocab_size=vocab_size).to(config["device"])
    model_2 = GPT_v2(vocab_size=vocab_size, n_embd=32).to(config["device"])
    model_3 = GPT_v3(vocab_size=vocab_size, n_embd=32, block_size=8, device=config["device"]).to(config["device"])

    save_model_info(model_1, input_tensor=dummy_input, folder=save_folder)


    print("Generate Text from Undtrained Model...")
    # Generate Text from Undtrained Model
    context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])  # Starting context for text generation
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))
    
    print("Training the model...")
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    trainer = Trainer(data=batch_generator, model=model, optimizer=optimizer, loss_fn=torch.nn.functional.cross_entropy)

    # Step 5: Train the Model
    trainer.train(config["batch_size"], config["max_iters"], config["eval_interval"])

    # Step 6: Generate Text from the Trained Model
    context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])  # Starting context for text generation
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))

    # training_loop(GPT_v1)

    # training_loop(GPT_v2, model_args={"n_embd":32})

    # training_loop(GPT_v3, model_args={"n_embd":32, "block_size": 8, "device": config["device"]})





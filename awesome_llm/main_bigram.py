from pathlib import Path
from awesome_llm.optimizers.adamw import AdamW
from awesome_llm.trainers.trainer import Trainer
from awesome_llm.utils.torch import save_model_info
from data.text_data import TextDataLoader, CharacterTokenizer, TextBatchGenerator
from utils.data import split_data
import torch

# Define file path for the dataset
file_path = Path(__file__).parent / 'tiny-shakespeare.txt'

# Set hyperparameters
train_test_split = 0.9
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2

# Determine the computation device based on CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training_loop(model_class, model_args={}, hyperparameters={}):

    # Override default hyperparameters with any provided values
    ....

    # Step 1: Load and Tokenize Data
    data_loader = TextDataLoader()
    raw_text = data_loader.load_data(file_path)
    tokenizer = CharacterTokenizer(raw_text)
    tokenized_text = tokenizer.tokenize(raw_text)

    # Step 2: Split Data into Training, Validation, and Testing Sets
    train_data, val_data, test_data = split_data(tokenized_text, train_pct=0.7, dev_pct=0.15)

    # Step 3: Initialize Batch Generator
    data_splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    batch_generator = TextBatchGenerator(data_splits, block_size=block_size)

    # Step 4: Initialize Model, Optimizer, and Trainer
    vocab_size = tokenizer.vocab_size()
    

    model = model_class(**model_args)
    save_model_info(model, input_tensor=torch.zeros((1, 1), dtype=torch.long, device=device), folder=Path(__file__).parent)

    

    save_model_info(model, input_tensor=torch.zeros((1, 1), dtype=torch.long, device=device), folder=Path(__file__).parent)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(data=batch_generator, model=model, optimizer=optimizer, loss_fn=torch.nn.functional.cross_entropy)

    # Step 5: Train the Model
    trainer.train(batch_size, max_iters, eval_interval)

    # Step 6: Generate Text from the Model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting context for text generation
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))



def model_version_2():
    # Step 1: Load and Tokenize Data
    data_loader = TextDataLoader()
    raw_text = data_loader.load_data(file_path)
    tokenizer = CharacterTokenizer(raw_text)
    tokenized_text = tokenizer.tokenize(raw_text)

    # Step 2: Split Data into Training, Validation, and Testing Sets
    train_data, val_data, test_data = split_data(tokenized_text, train_pct=0.7, dev_pct=0.15)

    # Step 3: Initialize Batch Generator
    data_splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    batch_generator = TextBatchGenerator(data_splits, block_size=block_size)

    # Step 4: Initialize Model, Optimizer, and Trainer
    vocab_size = tokenizer.vocab_size()
    from awesome_llm.modules.model_v2 import GPT
    # Set the embedding size to 32
    model = GPT(vocab_size=vocab_size, n_embd=32)
    save_model_info(model, input_tensor=torch.zeros((1, 1), dtype=torch.long, device=device), folder=Path(__file__).parent)


    optimizer = AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(data=batch_generator, model=model, optimizer=optimizer, loss_fn=torch.nn.functional.cross_entropy)

    # Step 5: Train the Model
    trainer.train(batch_size, max_iters, eval_interval)

    # Step 6: Generate Text from the Model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting context for text generation
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))


if __name__ == "__main__":
    model_version_1()
    model_version_2()

# Author: Piergiuseppe Mallozzi
# Year: 2024


import torch
from utils.data import process_data

from awesome_llm.models.model_v1 import GPT_v1
from awesome_llm.models.model_v2 import GPT_v2
from awesome_llm.models.model_v3 import GPT_v3
from awesome_llm.models.model_v4 import GPT_v4
from awesome_llm.models.model_v5 import GPT_v5
from awesome_llm.models.model_v6 import GPT_v6
from awesome_llm.models.model_v7 import GPT_v7
from awesome_llm.utils.torch import save_model_info
from awesome_llm.utils.train import generate, train_model

data_path = Path(__file__).parent / "data" / "tiny-shakespeare.txt"
save_folder = Path(__file__).parent / "info"

# Hyperparameters for initial and scaled models
hyperparameters_small = {
    "train_val_split": 0.9,
    "batch_size": 32,
    "max_iters": 3000,
    "block_size": 64,
    "eval_interval": 300,
    "learning_rate": 1e-2,
}

hyperparameters_scaled = {
    "train_val_split": 0.9,
    "batch_size": 64,
    "block_size": 256,
    "max_iters": 5000,
    "eval_interval": 200,
    "learning_rate": 3e-4,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Adjust the process_data function call if necessary to accommodate
# hyperparameters
vocab_size, tokenizer, batch_generator = process_data(
    data_path, hyperparameters=hyperparameters_small
)

dummy_input = torch.zeros((1, 1), dtype=torch.long, device=device)

# Define your models with commentary on what each adds to the previous version.
models = {
    # GPT_v1: Basic model with a simple embedding layer, demonstrating the
    # foundational structure.
    "GPT_v1": GPT_v1(vocab_size=vocab_size),
    # GPT_v2: Adds a linear transformation layer on top of the embeddings,
    # introducing the ability to learn more complex mappings.
    "GPT_v2": GPT_v2(vocab_size=vocab_size, n_embd=32),
    # GPT_v3: Incorporates positional embeddings, allowing the model to
    # understand sequence order, crucial for text processing.
    "GPT_v3": GPT_v3(vocab_size=vocab_size, n_embd=32, block_size=8),
    # GPT_v4: Introduces a single head of self-attention, marking the
    # beginning of the model's capacity for contextual understanding.
    "GPT_v4": GPT_v4(
        vocab_size=vocab_size, n_embd=32, block_size=8, head_size=32, dropout=0.1
    ),
    # GPT_v5: Expands to multi-head attention, enabling the model to attend to
    # different parts of the sequence simultaneously.
    "GPT_v5": GPT_v5(
        vocab_size=vocab_size,
        n_embd=32,
        block_size=8,
        head_size=8,
        dropout=0.1,
        num_heads=4,
    ),
    # GPT_v6: Add a FeedForward (computation block)
    "GPT_v6": GPT_v6(
        vocab_size=vocab_size,
        n_embd=32,
        block_size=8,
        head_size=8,
        dropout=0.1,
        num_heads=4,
    ),
    # GPT_v7: Adds multiple layers of Transformer blocks, each block includes
    # MultiHeadAttention and FeedForward with residual connections and layer
    # normalization.
    "GPT_v7": GPT_v7(
        vocab_size=vocab_size,
        n_embd=32,
        block_size=8,
        dropout=0.1,
        num_heads=4,
        num_layers=3,
    ),
    # GPT_final: GPT_v7 scaled up
    "GPT_final": GPT_v7(
        vocab_size=vocab_size,
        n_embd=384,
        block_size=256,
        dropout=0.2,
        num_heads=6,
        num_layers=6,
    ),
}


def process_model(model_id: str, hyperparameters: dict):
    model = models[model_id]
    model.to(device)
    # Save architecture info
    save_model_info(model, input_tensor=dummy_input, folder=save_folder, id=model_id)
    # Generate text before training
    print(f"Generated text by {model_id} before training:")
    print(generate(model, context=dummy_input, tokenizer=tokenizer))
    # Train the model
    train_model(model, hyperparameters, batch_generator)
    # Generate text after training
    print(f"Generated text by {model_id} after training:")
    print(generate(model, context=dummy_input, tokenizer=tokenizer))


# Example usage
process_model("GPT_v6", hyperparameters_small)

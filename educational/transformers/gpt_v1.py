import torch
import torch.nn as nn
import toml
import torch.onnx
from torch.nn import functional as F
from pathlib import Path
from torchviz import make_dot
from torchinfo import summary
from torchview import draw_graph

from bertviz import head_view, model_view


def pretty_print_tensor(tensor: torch.Tensor, name: str = "Tensor", num_entries: int = 2):
    """
    Pretty prints information about a PyTorch tensor.

    :param tensor: The tensor to be printed.
    :param num_entries: The number of entries from the tensor to display (default is 10).
    """
    print("-------------------")
    print(f"{name} Information:")
    print(f"Shape: {tensor.shape}\tDatatype: {tensor.dtype}")  # Print the shape and datatype of the tensor
    print(f"Data: {tensor.tolist()[:num_entries]}...")  # Print the first few entries of the tensor followed by "..."  


device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)


# Load config from TOML file
config_path = Path(__file__).parent / 'config_tiny.toml'  # Update this path
config = toml.load(config_path)


# Assigning hyperparameters from config
batch_size = config["batch_size"]  # How many independent sequences to process at once?
block_size = config["block_size"]  # What is the maximum context length for predictions?
max_iters = config["max_iters"]    # How many training iterations to run?
eval_interval = config["eval_interval"]  # How often to evaluate the model on the validation set?
learning_rate = config["learning_rate"]  # Learning rate for Adam optimizer (found through trial and error)
eval_iters = config["eval_iters"]  # How many batches to use per loss evaluation?
n_embd = config["n_embd"]          # Number of hidden units in the Transformer (384/6 = 64 dimensions per head)
n_head = config["n_head"]          # Number of attention heads in a single Transformer layer
n_layer = config["n_layer"]        # Number of Transformer layers
dropout = config["dropout"]        # Dropout probability



torch.manual_seed(1337)

input_file = Path(__file__).parent / 'tiny-shakespeare.txt'
# Load Tiny Shakespeare dataset 
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# (also refer to Andrej Karpathy's blog: http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from characters to indices and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]          # encoder: Take a string, return a list of indices/integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: Take a list of indices/integers, return a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% of all characters are for training
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generates a tensor of shape (batch_size,) with random sequence start indices between 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix])       # Stack all (ix holds batch_size many) sequences of this batch row-wise on top of each other to form a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # Same as x but shifted by one token
    x, y = x.to(device), y.to(device)
    return x, y # x is batch_size x block_size, y is batch_size x block_size

@torch.no_grad() # Disable gradient calculation for this function
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

class Head(nn.Module):
    """
    Implements a single head of self-attention within a Transformer model.
    
    This component is responsible for calculating the self-attention mechanism for a subset of the model's attention heads.
    It transforms the input embeddings into key, query, and value vectors to compute attention scores, which are then used
    to create a weighted combination of values, reflecting the input's relevance at different positions.
    
    Args:
        head_size (int): Dimensionality of the output space for this attention head.
        n_embd (int): Dimensionality of the input embeddings.
        dropout_probability (float): Probability of an element to be zeroed in the dropout step.
        block_size (int): Maximum sequence length that this model can handle.
    """
    def __init__(self, head_size, n_embd=config["n_embd"], dropout_probability=config["dropout"], block_size=config["block_size"]):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)     # Token-specific "What am I looking for?" information
        self.query = nn.Linear(n_embd, head_size, bias=False)   # Token-specific "What do I contain?" information

        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Lower triangular matrix for masking, ensuring causality in attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        """
        Forward pass for computing one head of self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is batch size, T is sequence length, and C is the embedding dimension.
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention and weighted aggregation of values, of shape (B, T, head_size).
        """
        B, T, C = x.shape  # Batch size (B), sequence length (T), embedding dimension (C = n_embd)
        
        k = self.key(x)   # (B,T,C) -> (B,T, head_size), transforming input to key vectors

        q = self.query(x) # (B,T,C) -> (B,T, head_size), transforming input to query vectors

        v = self.value(x) # (B,T,C) -> (B,T, head_size), transforming input to value vectors
        
        # Calculate attention scores by dot product of queries and keys
        # This operation transforms the shapes as follows:
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T), computing the attention scores for each query-key pair. For every row of B, we are going to have a T x T matrix giving use the affinities
        attention_scores = q @ k.transpose(-2, -1) 

        # Scaling by C**-0.5 (inverse square root of the embedding dimension) for normalization
        # Used to control the variance, especially at initialization
        attention_scores = attention_scores * C**-0.5
        
        # Apply masking to the upper triangle of the attention score matrix, setting future tokens' attention to -inf
        # This ensures that the prediction for a position does not depend on future tokens, maintaining causality.
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax applied to convert scores to probabilities, ensuring that the scores sum up to 1 across the sequence
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Dropout is applied to the attention probabilities to introduce randomness, helping prevent overfitting
        attention_probs = self.dropout(attention_probs)
        
        # Weighted sum of the values based on the computed attention probabilities
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size), aggregating information across the sequence based on attention
        attention_output = attention_probs @ v
        
        return attention_output

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Create num_heads many heads
        self.proj = nn.Linear(n_embd, n_embd)                                   # Projecting back to n_embd dimensions (the original size of the input, because we use residual connections)
        self.dropout = nn.Dropout(dropout)                                      # Dropout layer for regularization

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the outputs of all heads
        out = self.dropout(self.proj(out))                  # Project back to n_embd dimensions (because we use residual connections) and apply dropout
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Linear layer with 4*n_embd outputs (AIAYN suggests 4*n_embd for residual connections as channel size)
            nn.ReLU(),                     # ReLU introduces non-linearity
            nn.Linear(4 * n_embd, n_embd), # Linear layer with n_embd outputs
            nn.Dropout(dropout),           # Dropout layer for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head                    # Adapting the head size to the number of heads
        self.sa = MultiHeadAttention(n_head, head_size) # Self-attention multi-head layer (the communication)
        self.ffwd = FeedFoward(n_embd)                  # Feed-forward so that the output has the same dimension as the input (the computation)
        self.ln1 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the self-attention layer)
        self.ln2 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the feed-forward layer)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))                    # Residual connection, forking off to the self-attention layer, LayerNorm is applied before the self-attention layer
        x = x + self.ffwd(self.ln2(x))                  # Residual connection, forking off to the feed-forward layer, LayerNorm is again applied before the feed-forward layer
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embd = nn.Embedding(vocab_size, n_embd)                                   # Embedding the vocabulary, each individual token is represented by a vector of size vocab_size x n_embd
        self.position_embd = nn.Embedding(block_size, n_embd)                                # Embedding the position, each position is represented by a vector of size block_size x n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)                                         # Linear layer to map the embedding to the vocabulary size

    def forward(self, idx, targets=None):
        # pretty_print_tensor(idx)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embd(idx)                                                      # Embedding the input, shape is (batch_size, block_size, n_embd) (B, T, n_embd)
        pos_embd = self.position_embd(torch.arange(T, device=device))                        # Embedding the position by providing an integer sequence up to block_size, shape is (block_size, n_embd) (T, n_embd)
        x = tok_embd + pos_embd                                                              # Adding the token embedding and the position embedding, shape is (batch_size, block_size, n_embd) (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                             # Calculating the logits, shape is (batch_size, block_size, vocab_size) (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                # Transpose logits to (B, C, T) (B=batch_size, T=block_size, C=vocab_size)
            targets = targets.view(B*T)                 # Transpose targets to (B, T)
            loss = F.cross_entropy(logits, targets)     # Calculating cross entropy loss across all tokens in the batch

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                    # Condition on the last block_size tokens (B, T)
            logits, _ = self(idx_cond)                         # Forward pass (this is the forward function) with the current sequence of characters idx, results in (B, T, C)
            logits = logits[:, -1, :]                          # Focus on the last token from the logits (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # Calculate the set of probabilities for the next token based on this last token, results in (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token (B, 1), the token with the highest probability is sampled most likely
            idx = torch.cat((idx, idx_next), dim=1)            # Add the new token to the sequence (B, T+1) for the next iteration
        return idx

# Model
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # print the number of parameters in the model


print("Model architecture:")
# Assume 'model' is your PyTorch model and it is already loaded or defined
model.eval()


dummy_input = torch.zeros((1, 1), dtype=torch.long, device=device)

# Define the path for the ONNX file
onnx_path = Path(__file__).parent / 'model.onnx' 

# Export the model
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  onnx_path,           # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

print("Architecture exported to ONNX")


print("Tochviz make_dot")
y = model(dummy_input)

dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=False, show_saved=True)
dot.render('_model_graph', format='pdf')  # This saves the graph as 'model_graph.png' in the current directory



# # Generate text from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)     # Start with single token as context
# print(decode(m.generate(context, max_new_tokens=5)[0].tolist()))


print("\n\nGenerate Summary")
summary(model, input_data=torch.zeros((1, 1), dtype=torch.long, device=device))


print("\nDrew Graph")
model_graph = draw_graph(model, 
                            input_data=torch.zeros((1, 1), 
                            dtype=torch.long, device=device), 
                            device=device, 
                            expand_nested=True, 
                            hide_inner_tensors=True,
                            hide_module_functions=False,
                            roll=False,
                            depth=20)
graph = model_graph.visual_graph
graph.render('_model_layers', format='pdf') 

# Create a PyTorch optimizer
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    print(f"Iter {iter}") # print the number of parameters in the model
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')     # Get batch
    logits, loss = model(xb, yb)    # Forward pass
    opt.zero_grad(set_to_none=True) # Reset gradients
    loss.backward()                 # Backward pass
    opt.step()                      # Update parameters
    # Save the model architecture
    if iter % 1000 == 0:
        torch.save(model, f"model_{iter}.pt")


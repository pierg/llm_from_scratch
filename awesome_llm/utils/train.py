from pathlib import Path
import torch
from awesome_llm.optimizers.adamw import AdamW
from awesome_llm.trainers.trainer import Trainer
from awesome_llm.utils.torch import save_model_info
from data.text_data import TextDataLoader, CharacterTokenizer, TextBatchGenerator
from utils.data import split_data

# Default Constants and Hyperparameters
DEFAULTS = {
    'FILE_PATH': Path(__file__).parent / 'tiny-shakespeare.txt',
    'BATCH_SIZE': 32,
    'BLOCK_SIZE': 8,
    'MAX_ITERS': 3000,
    'EVAL_INTERVAL': 300,
    'LEARNING_RATE': 1e-2,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_and_tokenize_data(file_path):
    data_loader = TextDataLoader()
    raw_text = data_loader.load_data(file_path)
    tokenizer = CharacterTokenizer(raw_text)
    tokenized_text = tokenizer.tokenize(raw_text)
    return tokenizer, tokenized_text

def split_data_sets(tokenized_text, train_pct=0.7, dev_pct=0.15):
    return split_data(tokenized_text, train_pct=train_pct, dev_pct=dev_pct)

def initialize_batch_generator(data_splits, block_size):
    return TextBatchGenerator(data_splits, block_size=block_size)

def setup_model_and_trainer(model_class, vocab_size, learning_rate, n_embd=None):
    model_args = {'vocab_size': vocab_size}
    if n_embd is not None:
        model_args['n_embd'] = n_embd
        
    model = model_class(**model_args)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer

def train_and_generate_text(model, optimizer, tokenizer, batch_generator, device, batch_size, max_iters, eval_interval):
    trainer = Trainer(data=batch_generator, model=model, optimizer=optimizer, loss_fn=torch.nn.functional.cross_entropy)
    trainer.train(batch_size, max_iters, eval_interval)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_tokens))

def train_model(model_class, n_embd=None, **kwargs):

    file_path = kwargs.get('file_path', DEFAULTS['FILE_PATH'])
    batch_size = kwargs.get('batch_size', DEFAULTS['BATCH_SIZE'])
    block_size = kwargs.get('block_size', DEFAULTS['BLOCK_SIZE'])
    max_iters = kwargs.get('max_iters', DEFAULTS['MAX_ITERS'])
    eval_interval = kwargs.get('eval_interval', DEFAULTS['EVAL_INTERVAL'])
    learning_rate = kwargs.get('learning_rate', DEFAULTS['LEARNING_RATE'])
    device = kwargs.get('device', DEFAULTS['DEVICE'])


    tokenizer, tokenized_text = load_and_tokenize_data(file_path)
    data_splits = split_data_sets(tokenized_text)
    batch_generator = initialize_batch_generator({'train': data_splits[0], 'val': data_splits[1], 'test': data_splits[2]}, block_size)
    vocab_size = tokenizer.vocab_size()
    model, optimizer = setup_model_and_trainer(model_class, vocab_size, device, learning_rate, n_embd=n_embd)

    # Save Model Info
    save_model_info(model, 
                    input_tensor=torch.zeros((1, 1), dtype=torch.long, device=device), 
                    folder=Path(__file__).parent / "info")
    
    train_and_generate_text(model, optimizer, tokenizer, batch_generator, device, batch_size, max_iters, eval_interval)


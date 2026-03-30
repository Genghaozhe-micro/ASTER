# /aster-llama/train_llama.py

import torch
import os
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import numpy as np

import config_llama as config
from model_utils_llama import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter
from llama_trainer import ASTERTrainerLLaMA

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Main entry point for training ASTER-LLaMA."""
    set_seed(config.SEED)
    print("--- ASTER-LLaMA Training Script ---")

    print("Loading base LLaMA model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    model.to(config.DEVICE)

    print("Loading BoolQ dataset...")
    try:
        raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')
    except Exception:
        print("Could not reach Hugging Face Hub. Attempting to use local cache.")
        raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')

    def preprocess_function(examples):
        # Format: "Passage: {passage}\nQuestion: {question}\nAnswer:"
        texts = []
        for passage, question in zip(examples['passage'], examples['question']):
            text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            texts.append(text)
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors=None,
        )
        # BoolQ labels: True=1 (Yes), False=0 (No)
        tokenized['labels'] = [int(ans) for ans in examples['answer']]
        return tokenized

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True,
                                           remove_columns=raw_datasets['train'].column_names)
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)
    print(f"Dataset processed. Train size: {len(train_dataset)}, DataLoader created.")

    print("Initializing ASTER components (Scorer and Adapter)...")
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    # Force float32 for Scorer/Adapter to avoid bfloat16 NaN in RL computations
    aster_dtype = torch.float32
    print(f"ASTER dtype: {aster_dtype}, hidden_dim: {hidden_dim}, num_layers: {num_layers}")

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, aster_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, aster_dtype).to(config.DEVICE)

    optimizer = AdamW(list(scorer.parameters()) + list(adapter.parameters()), lr=config.LEARNING_RATE)

    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_llama_checkpoint.pt")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        scorer.load_state_dict(checkpoint['scorer_state_dict'])
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed successfully. Starting from epoch {start_epoch + 1}.")
    elif args.resume:
        print(f"WARNING: --resume specified, but no checkpoint found at {checkpoint_path}.")

    print("Setting up ASTER-LLaMA trainer...")
    aster_trainer = ASTERTrainerLLaMA(model, scorer, adapter, optimizer)

    print(f"Starting training from epoch {start_epoch + 1}...")
    aster_trainer.train(train_dataloader, tokenizer, start_epoch=start_epoch)

    print("--- Training finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ASTER framework on LLaMA.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint.")
    args = parser.parse_args()

    main(args)

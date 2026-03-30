# /aster-llama/train_llama_distributed.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import os
import argparse
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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_process(rank, world_size, args):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)
    set_seed(config.SEED)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print("Loading base LLaMA model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()

    # Rank 0 downloads and preprocesses first; others wait then read cache.
    def preprocess_function(examples):
        texts = []
        for passage, question in zip(examples['passage'], examples['question']):
            text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            texts.append(text)
        tokenized = tokenizer(texts, padding="max_length", truncation=True,
                              max_length=config.MAX_SEQ_LENGTH, return_tensors=None)
        tokenized['labels'] = [int(ans) for ans in examples['answer']]
        return tokenized

    if rank == 0:
        print("Loading and processing dataset...")
        raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True,
                                               remove_columns=raw_datasets['train'].column_names, num_proc=4)
        tokenized_datasets.set_format("torch")
    dist.barrier()
    if rank != 0:
        raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True,
                                               remove_columns=raw_datasets['train'].column_names, num_proc=4)
        tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"]

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler,
                                  num_workers=4, pin_memory=True, shuffle=False)
    if rank == 0:
        print("Distributed Sampler and DataLoader created.")

    if rank == 0:
        print("Initializing ASTER components...")
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    # Force float32 for Scorer/Adapter to avoid bfloat16 NaN in RL computations
    aster_dtype = torch.float32

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, aster_dtype).to(device)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, aster_dtype).to(device)

    scorer = DDP(scorer, device_ids=[rank], find_unused_parameters=True)
    adapter = DDP(adapter, device_ids=[rank], find_unused_parameters=True)

    optimizer = AdamW(list(scorer.parameters()) + list(adapter.parameters()),
                      lr=config.LEARNING_RATE * world_size)

    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_llama_checkpoint.pt")
    if args.resume and os.path.exists(checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        scorer.module.load_state_dict(checkpoint['scorer_state_dict'])
        adapter.module.load_state_dict(checkpoint['adapter_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resumed successfully from epoch {start_epoch}.")
    elif args.resume and rank == 0:
        print(f"WARNING: --resume specified, but no checkpoint found at {checkpoint_path}.")

    if rank == 0:
        print("Setting up ASTER-LLaMA trainer...")
    aster_trainer = ASTERTrainerLLaMA(model, scorer, adapter, optimizer, rank, world_size)

    if rank == 0:
        print(f"Starting distributed training from epoch {start_epoch + 1}...")
    aster_trainer.train(train_dataloader, tokenizer, train_sampler, start_epoch=start_epoch)

    if rank == 0:
        print("Training finished successfully!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASTER-LLaMA using Distributed Data Parallel.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs. Please use train_llama.py for single-GPU training.")
        exit()
    else:
        print(f"Found {world_size} GPUs. Starting distributed training...")
        mp.spawn(main_process, args=(world_size, args), nprocs=world_size, join=True)

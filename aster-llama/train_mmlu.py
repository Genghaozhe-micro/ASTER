# /aster-llama/train_mmlu.py
# Train ASTER on MMLU (4-choice QA) with Qwen2.5-7B, 8-GPU DDP

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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Use MMLU config
import config_mmlu as config
from model_utils_llama import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter

# Import trainer but override its config reference
from llama_trainer import ASTERTrainerLLaMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # different port from BoolQ
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_process(rank, world_size, args):
    print(f"Running MMLU DDP training on rank {rank}.")
    setup(rank, world_size)
    set_seed(config.SEED)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print("Loading Qwen2.5-7B...")
    model, tokenizer = load_model_and_tokenizer()
    model.to(device).eval()

    # MMLU choice token IDs
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A', 'B', 'C', 'D']]
    if rank == 0:
        print(f"Choice token IDs: {choice_ids}")

    # Preprocess function for MMLU
    def preprocess(examples):
        texts = []
        labels = []
        for q, choices, ans in zip(examples['question'], examples['choices'], examples['answer']):
            text = f"Question: {q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            texts.append(text)
            labels.append(ans)
        tok = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=config.MAX_SEQ_LENGTH, return_tensors=None)
        tok['labels'] = labels
        return tok

    # Dataset: rank 0 first, then barrier
    if rank == 0:
        print("Loading MMLU dataset...")
        raw = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, verification_mode='no_checks')
        # Use auxiliary_train as training set (much larger)
        train_raw = raw['auxiliary_train'] if 'auxiliary_train' in raw else raw['test']
        tok_ds = train_raw.map(preprocess, batched=True,
                               remove_columns=train_raw.column_names, num_proc=4)
        tok_ds.set_format("torch")
    dist.barrier()
    if rank != 0:
        raw = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, verification_mode='no_checks')
        train_raw = raw['auxiliary_train'] if 'auxiliary_train' in raw else raw['test']
        tok_ds = train_raw.map(preprocess, batched=True,
                               remove_columns=train_raw.column_names, num_proc=4)
        tok_ds.set_format("torch")

    sampler = DistributedSampler(tok_ds, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(tok_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                            num_workers=4, pin_memory=True)
    if rank == 0:
        print(f"Train size: {len(tok_ds)}, DataLoader created.")

    # ASTER components (float32)
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, torch.float32).to(device)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, torch.float32).to(device)
    scorer = DDP(scorer, device_ids=[rank], find_unused_parameters=True)
    adapter = DDP(adapter, device_ids=[rank], find_unused_parameters=True)
    optimizer = AdamW(list(scorer.parameters()) + list(adapter.parameters()),
                      lr=config.LEARNING_RATE * world_size)

    # Resume
    start_epoch = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "aster_mmlu_checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location={'cuda:0': f'cuda:{rank}'})
        scorer.module.load_state_dict(ckpt['scorer_state_dict'])
        adapter.module.load_state_dict(ckpt['adapter_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}.")

    # Trainer — override config to use MMLU config
    trainer = ASTERTrainerLLaMA(model, scorer, adapter, optimizer, rank, world_size)
    trainer.config = config  # point to MMLU config

    # Override checkpoint path
    original_save = trainer.save_checkpoint
    def save_checkpoint_mmlu(epoch):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_mmlu_checkpoint.pt")
        scorer_sd = trainer.scorer.module.state_dict() if trainer.world_size > 1 else trainer.scorer.state_dict()
        adapter_sd = trainer.adapter.module.state_dict() if trainer.world_size > 1 else trainer.adapter.state_dict()
        torch.save({'epoch': epoch, 'scorer_state_dict': scorer_sd,
                     'adapter_state_dict': adapter_sd,
                     'optimizer_state_dict': trainer.optimizer.state_dict()}, checkpoint_path)
        print(f"Saved MMLU checkpoint for epoch {epoch + 1} to {checkpoint_path}")
    trainer.save_checkpoint = save_checkpoint_mmlu

    # Need to override _get_final_logits_and_pred for MMLU (4-choice via A/B/C/D logits)
    # and pass choice_ids through tokenizer
    # The trainer's train() expects yes/no tokens; we pass A/B/C/D as no_token/yes_token equivalent
    # Simpler: override the train call with MMLU-specific logic
    if rank == 0:
        print(f"Starting MMLU training from epoch {start_epoch + 1}...")
    trainer.train_mmlu(dataloader, tokenizer, choice_ids, sampler, start_epoch)

    if rank == 0:
        print("MMLU Training finished!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need >= 2 GPUs.")
        exit()
    print(f"Found {world_size} GPUs. Starting MMLU distributed training...")
    mp.spawn(main_process, args=(world_size, args), nprocs=world_size, join=True)

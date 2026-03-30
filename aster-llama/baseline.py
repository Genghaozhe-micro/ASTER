# /aster-llama/baseline.py
# Evaluate the full LLaMA model on BoolQ without any layer skipping.

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from tqdm import tqdm
import argparse
import numpy as np
import random

import config_llama as config
from model_utils_llama import load_model_and_tokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_baseline_evaluation(args):
    set_seed(config.SEED)
    print(f"--- Starting Baseline Evaluation (Full Model) ---")
    print(f"--- Random Seed set to: {config.SEED} ---")

    print(f"Loading original pre-trained model '{config.MODEL_ID}' for baseline test.")
    model, tokenizer = load_model_and_tokenizer()
    model.to(config.DEVICE)
    model.eval()

    print(f"Loading evaluation dataset ({config.DATASET_NAME})...")
    raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')

    def preprocess_function(examples):
        texts = []
        for passage, question in zip(examples['passage'], examples['question']):
            text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            texts.append(text)
        tokenized = tokenizer(texts, padding="max_length", truncation=True,
                              max_length=config.MAX_SEQ_LENGTH, return_tensors=None)
        tokenized['labels'] = [int(ans) for ans in examples['answer']]
        return tokenized

    eval_dataset = raw_datasets["validation"].map(preprocess_function, batched=True,
                                                    remove_columns=raw_datasets['validation'].column_names)
    eval_dataset.set_format("torch")
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Precompute yes/no token IDs
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {os.path.basename(config.MODEL_ID)}"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, seq_len, vocab_size)

            # Get logits at the last non-padding token
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
            batch_size = input_ids.shape[0]
            last_logits = logits[torch.arange(batch_size, device=logits.device), last_token_idx]  # (B, vocab)

            # Binary: compare Yes vs No logits
            binary_logits = torch.stack([last_logits[:, no_token_id], last_logits[:, yes_token_id]], dim=-1)
            predictions = torch.argmax(binary_logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print("\n" + "=" * 60)
    print("--- Model Baseline Evaluation Final Results ---")
    print(f"  Evaluated Model: {config.MODEL_ID} (Full, no skipping)")
    print(f"  Evaluation Dataset: {config.DATASET_NAME} (validation split)")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline LLaMA model without layer skipping.")
    args = parser.parse_args()
    run_baseline_evaluation(args)

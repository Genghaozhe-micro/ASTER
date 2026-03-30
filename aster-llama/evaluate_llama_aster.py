# /aster-llama/evaluate_llama_aster.py

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
import os
from tqdm import tqdm
import random
import numpy as np

import config_llama as config
from model_utils_llama import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter

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


def predict_greedy(model, scorer, adapter, input_ids, attention_mask):
    """
    Greedy inference: at each decision step, pick the candidate layer with the highest score.
    Uses last token as cognitive token (decoder-only model).
    """
    num_layers = model.config.num_hidden_layers
    llama_layers = model.model.layers
    batch_size = input_ids.shape[0]

    # Find last non-padding token index
    last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)

    with torch.no_grad():
        student_hidden_state = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = model.model.rotary_emb(student_hidden_state, position_ids)

        l_curr = 0
        executed_layers_count = 0

        while l_curr < num_layers - 1:
            executed_layers_count += 1

            # Execute current layer
            layer_output = llama_layers[l_curr](
                student_hidden_state,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )
            student_hidden_state = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Extract cognitive token (last token)
            cog_state = student_hidden_state[
                torch.arange(batch_size, device=student_hidden_state.device), last_token_idx
            ]

            # Determine candidates
            required_future_steps = config.MIN_TOTAL_EXECUTED_LAYERS - executed_layers_count
            if required_future_steps > 0:
                max_allowed_l_next = num_layers - required_future_steps
            else:
                max_allowed_l_next = num_layers

            start_candidate = l_curr + 1
            end_candidate = min(num_layers, max_allowed_l_next + 1)
            candidate_layers = list(range(start_candidate, end_candidate))

            if not candidate_layers and start_candidate < num_layers:
                candidate_layers = [start_candidate]

            if not candidate_layers:
                break

            # Greedy: pick argmax (float32 for scorer)
            scores = scorer(h_cog=cog_state.float(), l_curr=l_curr, candidate_layers=candidate_layers)
            action_index = torch.argmax(scores, dim=-1).item()
            l_next = candidate_layers[action_index]

            if l_next > l_curr + 1:
                orig_dtype = student_hidden_state.dtype
                student_hidden_state = adapter(student_hidden_state.float(), l_curr, l_next).to(orig_dtype)

            l_curr = l_next

        # Execute remaining layers
        for final_l in range(l_curr, num_layers):
            layer_output = llama_layers[final_l](
                student_hidden_state,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )
            student_hidden_state = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            executed_layers_count += 1

        # Get prediction: norm → lm_head → yes/no
        normed = model.model.norm(student_hidden_state)
        last_hidden = normed[torch.arange(batch_size, device=normed.device), last_token_idx]
        logits = model.lm_head(last_hidden)

    return logits, executed_layers_count


def run_evaluation(args):
    print("--- Starting ASTER-LLaMA Evaluation ---")
    set_seed(config.SEED)

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_llama_checkpoint.pt")
        print(f"Checkpoint path not provided, using default: {args.checkpoint_path}")

    if not os.path.exists(args.checkpoint_path):
        print(f"[FATAL ERROR] Checkpoint file not found at {args.checkpoint_path}")
        return

    model, tokenizer = load_model_and_tokenizer()
    model.to(config.DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)

    print(f"Loading trained weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=config.DEVICE)
    scorer.load_state_dict(checkpoint['scorer_state_dict'])
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    print("Successfully loaded Scorer and Adapter weights.")
    scorer.eval()
    adapter.eval()

    print(f"Loading evaluation dataset ({config.DATASET_NAME})...")
    try:
        raw_datasets = load_dataset(config.DATASET_NAME, verification_mode='no_checks')
    except Exception:
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
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Precompute yes/no token IDs
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    total_correct, total_executed_layers, total_samples = 0, 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating with ASTER-LLaMA"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        label = batch['labels'].item()

        logits, executed_layers = predict_greedy(model, scorer, adapter, input_ids, attention_mask)

        binary_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
        prediction = torch.argmax(binary_logits, dim=-1).item()

        if prediction == label:
            total_correct += 1
        total_executed_layers += executed_layers
        total_samples += 1

    if total_samples == 0:
        print("No samples were evaluated.")
        return

    accuracy = total_correct / total_samples
    avg_executed_layers = total_executed_layers / total_samples
    speedup_ratio = num_layers / avg_executed_layers

    print("\n" + "=" * 60)
    print("--- ASTER-LLaMA Evaluation Final Results ---")
    print(f"  Evaluated Checkpoint: {args.checkpoint_path}")
    print(f"  Evaluation Dataset: {config.DATASET_NAME} (validation split)")
    print(f"  Decision Strategy: Greedy Search")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"  Average Executed Layers: {avg_executed_layers:.2f} / {num_layers}")
    print(f"  Computational Savings: {100 * (1 - avg_executed_layers / num_layers):.2f}%")
    print(f"  Effective Speedup vs Full Model: {speedup_ratio:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ASTER-LLaMA model.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Full path to the saved checkpoint file.")
    args = parser.parse_args()

    run_evaluation(args)

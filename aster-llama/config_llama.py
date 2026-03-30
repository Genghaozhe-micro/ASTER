# /aster-llama/config_llama.py

import torch

SEED = 42
# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Configuration ---
# Qwen2.5-7B as a drop-in replacement for LLaMA 3.1-8B (same decoder-only architecture)
MODEL_ID = 'Qwen/Qwen2.5-7B'

# --- Dataset Configuration ---
# BoolQ: passage + question → yes/no (binary classification)
DATASET_NAME = "google/boolq"
DATASET_CONFIG = None

# --- Checkpoint and Logging Configuration ---
CHECKPOINT_DIR = "./checkpoints_qwen_boolq"
EVAL_OUTPUT_DIR = "./evaluation_results_qwen_boolq"

# --- Training Hyperparameters ---
NUM_EPOCHS = 3
BATCH_SIZE = 2          # LLaMA-8B is much larger; use small batch + gradient accumulation
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
CLIP_GRAD_NORM = 1.0
LOG_INTERVAL = 10
MAX_SEQ_LENGTH = 256    # BoolQ passages can be long

# --- ASTER Component Dimensions ---
# Qwen2.5-7B hidden_dim = 3584
ADAPTER_BOTTLENECK_DIM = 256
SCORER_HIDDEN_DIM = 512

# --- Loss Composition Weights (Lambdas) ---
CE_LOSS_WEIGHT = 10
RL_LOSS_WEIGHT = 0.5
KD_LOSS_WEIGHT = 0.001

# --- Reinforcement Learning (TIDR) Hyperparameters ---
GAMMA = 0.99
W_EFFICIENCY = 0.01
W_TASK = 1.0
SKIP_PENALTY_WEIGHT = 0.1

# --- Knowledge Distillation Hyperparameters ---
KD_TEMP = 1.0

# Qwen2.5-7B has 28 layers; require at least 7 executed layers to preserve quality
MIN_TOTAL_EXECUTED_LAYERS = 7

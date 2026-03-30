# /aster-llama/config_mmlu.py

import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
MODEL_ID = 'Qwen/Qwen2.5-7B'

# --- Dataset ---
DATASET_NAME = "cais/mmlu"
DATASET_CONFIG = "all"

# --- Checkpoint ---
CHECKPOINT_DIR = "./checkpoints_qwen_mmlu"

# --- Training ---
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
CLIP_GRAD_NORM = 1.0
LOG_INTERVAL = 10
MAX_SEQ_LENGTH = 256

# --- ASTER Components ---
ADAPTER_BOTTLENECK_DIM = 256
SCORER_HIDDEN_DIM = 512

# --- Loss Weights ---
CE_LOSS_WEIGHT = 10
RL_LOSS_WEIGHT = 0.5
KD_LOSS_WEIGHT = 0.001

# --- RL ---
GAMMA = 0.99
W_EFFICIENCY = 0.01
W_TASK = 1.0
SKIP_PENALTY_WEIGHT = 0.1

# --- KD ---
KD_TEMP = 1.0

MIN_TOTAL_EXECUTED_LAYERS = 7

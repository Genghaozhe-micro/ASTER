# /aster-llama/model_utils_llama.py

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import config_llama as config

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_model_and_tokenizer():
    """
    Loads the LLaMA model and tokenizer.
    Model is loaded to CPU first; the caller moves it to the correct device.
    Uses bfloat16 to fit LLaMA-8B in GPU memory.
    """
    print(f"--- Loading Model and Tokenizer for {config.MODEL_ID} ---")

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype="auto",
        output_hidden_states=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    # LLaMA tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Hidden dim: {model.config.hidden_size}, "
          f"Dtype: {next(model.parameters()).dtype}")
    return model, tokenizer

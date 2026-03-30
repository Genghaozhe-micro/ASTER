# /aster-llama/components.py

import torch
import torch.nn as nn

from config_llama import DEVICE


class DynamicAdapter(nn.Module):
    """
    Compensates for information loss when layers are skipped.
    Identical architecture to the BERT version — architecture-agnostic.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int, num_layers: int, dtype: torch.dtype):
        super().__init__()
        self.layer_embedding = nn.Embedding(
            num_embeddings=num_layers,
            embedding_dim=hidden_dim
        ).to(device=DEVICE, dtype=dtype)

        self.adapter_network = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim, device=DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim, device=DEVICE, dtype=dtype)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, device=DEVICE, dtype=dtype)

    def forward(self, hidden_state: torch.Tensor, l_curr: int, l_next: int) -> torch.Tensor:
        residual = hidden_state
        l_curr_tensor = torch.tensor([l_curr], device=hidden_state.device)
        l_next_tensor = torch.tensor([l_next], device=hidden_state.device)

        curr_layer_emb = self.layer_embedding(l_curr_tensor)
        next_layer_emb = self.layer_embedding(l_next_tensor)

        skip_signal = next_layer_emb - curr_layer_emb
        conditioned_state = hidden_state + skip_signal.unsqueeze(1).to(hidden_state.dtype)
        adapter_output = self.adapter_network(conditioned_state)

        output_state = self.layer_norm(residual + adapter_output)
        return output_state


class ScoringModel(nn.Module):
    """
    Policy network (Scorer) for the MDP.
    For LLaMA (decoder-only), uses the LAST token's hidden state as the cognitive token,
    as described in §3.2 of the paper.
    """
    def __init__(self, hidden_dim: int, scorer_hidden_dim: int, num_layers: int, dtype: torch.dtype):
        super().__init__()
        embedding_dim_scorer = hidden_dim // 4
        self.layer_embedding = nn.Embedding(
            num_embeddings=num_layers,
            embedding_dim=embedding_dim_scorer
        ).to(device=DEVICE, dtype=dtype)

        self.scorer_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim_scorer * 2, scorer_hidden_dim, device=DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1, device=DEVICE, dtype=dtype)
        )

    def forward(self, h_cog: torch.Tensor, l_curr: int, candidate_layers: list) -> torch.Tensor:
        """
        Args:
            h_cog: Cognitive token representation (last token), shape (B, D)
            l_curr: Current layer index
            candidate_layers: List of candidate next-layer indices
        Returns:
            scores: (B, num_candidates)
        """
        batch_size = h_cog.shape[0]
        l_curr_tensor = torch.tensor([l_curr] * batch_size, device=h_cog.device)
        scores = []
        for l_cand in candidate_layers:
            l_cand_tensor = torch.tensor([l_cand] * batch_size, device=h_cog.device)
            l_curr_emb = self.layer_embedding(l_curr_tensor)
            l_cand_emb = self.layer_embedding(l_cand_tensor)
            mlp_input = torch.cat([h_cog, l_curr_emb, l_cand_emb], dim=-1)
            score = self.scorer_mlp(mlp_input)
            scores.append(score)
        return torch.cat(scores, dim=-1)

# /aster-llama/llama_trainer.py

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import wandb

from reward import TIDR_V2
from components import ScoringModel, DynamicAdapter
import config_llama as config


class ASTERTrainerLLaMA:
    def __init__(self, model, scorer, adapter, optimizer, rank=0, world_size=1):
        self.model = model
        self.scorer = scorer
        self.adapter = adapter
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(f'cuda:{rank}')
        self.rank = rank
        self.world_size = world_size

        # LLaMA architecture: model.model.layers is the list of decoder layers
        self.llama_layers = model.model.layers
        self.num_layers = len(self.llama_layers)

        self.reward_fn = TIDR_V2(
            w_task=config.W_TASK,
            w_efficiency=config.W_EFFICIENCY,
            total_layers=self.num_layers,
            device=self.device,
            skip_penalty_weight=config.SKIP_PENALTY_WEIGHT
        )

        # Logging: only rank 0 writes logs
        self.writer = None
        if self.rank == 0:
            log_dir = os.path.join(config.CHECKPOINT_DIR, "tb_logs")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs → {log_dir}")

            # Initialize wandb
            wandb.init(
                project="aster-llama",
                name=f"qwen2.5-7b-boolq-{world_size}gpu",
                config={
                    "model": config.MODEL_ID,
                    "dataset": config.DATASET_NAME,
                    "num_layers": self.num_layers,
                    "hidden_dim": model.config.hidden_size,
                    "batch_size": config.BATCH_SIZE,
                    "grad_accum": config.GRADIENT_ACCUMULATION_STEPS,
                    "effective_batch": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS * world_size,
                    "lr": config.LEARNING_RATE * world_size,
                    "epochs": config.NUM_EPOCHS,
                    "min_executed_layers": config.MIN_TOTAL_EXECUTED_LAYERS,
                    "ce_weight": config.CE_LOSS_WEIGHT,
                    "rl_weight": config.RL_LOSS_WEIGHT,
                    "kd_weight": config.KD_LOSS_WEIGHT,
                    "world_size": world_size,
                },
            )
            print("Wandb initialized.")

    def _get_last_token_indices(self, attention_mask):
        """
        For each sample in the batch, find the index of the last non-padding token.
        This token serves as the cognitive token for decoder-only models (paper §3.2).
        """
        # attention_mask: (B, seq_len), 1 for real tokens, 0 for padding
        # sum of attention_mask per row gives the length; subtract 1 for 0-based index
        return attention_mask.sum(dim=1) - 1  # (B,)

    def _get_cognitive_token(self, hidden_state, last_token_idx):
        """
        Extract the last (cognitive) token's hidden state for each sample.
        For LLaMA, the cognitive token is the last token (paper §3.2, eq 4-5).
        """
        # hidden_state: (B, seq_len, D)
        batch_size = hidden_state.shape[0]
        return hidden_state[torch.arange(batch_size, device=hidden_state.device), last_token_idx]  # (B, D)

    def _get_final_logits_and_pred(self, final_hidden_state, last_token_idx):
        """
        Get logits from the LM head for the last token position, then map to yes/no.
        For BoolQ: compare logits for 'Yes' vs 'No' tokens.
        """
        # Apply RMSNorm (model.model.norm) then lm_head
        normed = self.model.model.norm(final_hidden_state)
        # Get logits at the last token position
        batch_size = final_hidden_state.shape[0]
        last_token_hidden = normed[torch.arange(batch_size, device=final_hidden_state.device), last_token_idx]
        logits = self.model.lm_head(last_token_hidden)  # (B, vocab_size)
        return logits

    def _compute_knowledge_distillation_loss(self, student_state, teacher_state,
                                              teacher_attn_weights, last_token_idx):
        """
        LLaMA KD loss (paper §3.6, eq 16):
        I_LLaMA(H_S, H_T, i) = λ_token * Softmax( (1/N_h) * Σ_h A_T^(l,h)[:, N] )_i

        The importance of each token is derived from the teacher's self-attention
        distribution toward the final token of the sequence.
        """
        batch_size = student_state.shape[0]
        seq_len = student_state.shape[1]

        if teacher_attn_weights is not None:
            # teacher_attn_weights: (B, num_heads, seq_len, seq_len)
            # Extract attention scores toward the last token: A[:, :, :, last_pos]
            # Then average across heads
            last_idx = last_token_idx.view(batch_size, 1, 1, 1).expand(-1, teacher_attn_weights.shape[1], seq_len, 1)
            attn_to_last = teacher_attn_weights.gather(3, last_idx).squeeze(-1)  # (B, num_heads, seq_len)
            avg_attn_to_last = attn_to_last.mean(dim=1)  # (B, seq_len)
            importance_weights = F.softmax(avg_attn_to_last, dim=-1).detach()  # (B, seq_len)
        else:
            # Fallback: uniform weights
            importance_weights = torch.ones(batch_size, seq_len, device=student_state.device) / seq_len

        # Token-level MSE weighted by importance
        mse_per_token = F.mse_loss(student_state, teacher_state, reduction='none').mean(dim=2)  # (B, seq_len)
        weighted_loss = (importance_weights * mse_per_token).sum(dim=1)  # (B,)

        return weighted_loss

    def _run_single_layer(self, hidden_state, layer_idx, attention_mask, position_embeddings):
        """Execute a single decoder layer (compatible with Qwen2/LLaMA)."""
        layer = self.llama_layers[layer_idx]
        layer_output = layer(
            hidden_state,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=False,
        )
        # layer_output is a tuple: (hidden_states, ...) or just hidden_states
        return layer_output[0] if isinstance(layer_output, tuple) else layer_output

    def _prepare_causal_mask(self, attention_mask):
        """
        Prepare the 4D causal attention mask for LLaMA.
        """
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        batch_size, seq_len = attention_mask.shape
        # Create causal mask
        causal_mask = AttentionMaskConverter._make_causal_mask(
            (batch_size, seq_len),
            dtype=torch.float32,
            device=attention_mask.device,
        )
        # Expand padding mask
        expanded_mask = AttentionMaskConverter._expand_mask(
            attention_mask, dtype=torch.float32, tgt_len=seq_len
        ).to(attention_mask.device)
        causal_mask = causal_mask + expanded_mask
        return causal_mask

    def train(self, train_dataloader, tokenizer, train_sampler=None, start_epoch=0):
        if self.rank == 0:
            print("--- Starting ASTER-LLaMA Training ---")

        # Precompute yes/no token IDs for BoolQ evaluation
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

        if start_epoch == 0:
            self.optimizer.zero_grad()
        global_step = start_epoch * len(train_dataloader)

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            if self.rank == 0:
                print(f"\n--- Epoch {epoch + 1}/{self.config.NUM_EPOCHS} ---")

            if self.world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(self.rank != 0))
            epoch_correct, epoch_total = 0, 0

            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size = input_ids.shape[0]

                last_token_idx = self._get_last_token_indices(attention_mask)

                # --- Teacher forward pass (full model, frozen) ---
                with torch.no_grad():
                    teacher_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=False,  # SDPA does not support output_attentions; use uniform KD weights
                    )
                    teacher_hidden_states = teacher_outputs.hidden_states  # (num_layers+1, B, seq, D)
                    teacher_attentions = teacher_outputs.attentions if teacher_outputs.attentions else None

                # --- Student dynamic path ---
                # Get embeddings
                student_hidden_state = self.model.model.embed_tokens(input_ids)

                # Create position IDs and compute rotary position embeddings (cos, sin)
                position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.model.model.rotary_emb(student_hidden_state, position_ids)

                # Prepare causal attention mask for layer calls
                causal_mask = None

                log_probs_rollout, kd_loss_rollout = [], []
                l_curr_rollout, l_next_rollout = [], []
                l_curr_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                executed_layers_count = 0

                while True:
                    active_mask = l_curr_batch < self.num_layers - 1
                    if not torch.any(active_mask):
                        break

                    executed_layers_count += 1
                    l_curr_rollout.append(l_curr_batch.clone())
                    next_hidden_state = student_hidden_state.clone()
                    step_log_probs = torch.zeros(batch_size, device=self.device)
                    step_kd_loss = torch.zeros(batch_size, device=self.device)
                    unique_l_currs = torch.unique(l_curr_batch[active_mask])

                    for l_c in unique_l_currs:
                        l_curr = l_c.item()
                        group_indices = (l_curr_batch == l_curr).nonzero(as_tuple=True)[0]
                        group_hidden_states = student_hidden_state.index_select(0, group_indices)
                        group_attention_mask = attention_mask.index_select(0, group_indices)
                        group_pos_emb = (position_embeddings[0].index_select(0, group_indices),
                                         position_embeddings[1].index_select(0, group_indices))

                        # Execute current layer
                        processed_states = self._run_single_layer(
                            group_hidden_states, l_curr, causal_mask, group_pos_emb
                        )

                        # KD loss: compare student output with teacher's next layer state (float32 for stability)
                        teacher_ref_state = teacher_hidden_states[l_curr + 1].detach().index_select(0, group_indices)
                        teacher_attn = teacher_attentions[l_curr].detach().index_select(0, group_indices) \
                            if teacher_attentions is not None else None
                        group_last_idx = last_token_idx.index_select(0, group_indices)
                        kd_loss = self._compute_knowledge_distillation_loss(
                            processed_states.float(), teacher_ref_state.float(), teacher_attn, group_last_idx
                        )

                        # Cognitive token = last token (paper §3.2)
                        cog_state = self._get_cognitive_token(processed_states, group_last_idx)

                        # Determine candidate layers
                        required_future_steps = self.config.MIN_TOTAL_EXECUTED_LAYERS - executed_layers_count
                        max_allowed_l_next = self.num_layers - required_future_steps \
                            if required_future_steps > 0 else self.num_layers
                        start_candidate = l_curr + 1
                        end_candidate = min(self.num_layers, max_allowed_l_next + 1)
                        candidate_layers = list(range(start_candidate, end_candidate))
                        if not candidate_layers and start_candidate < self.num_layers:
                            candidate_layers = [start_candidate]

                        if not candidate_layers:
                            l_curr_batch.index_fill_(0, group_indices, self.num_layers)
                            next_hidden_state.index_copy_(0, group_indices, processed_states)
                            continue

                        # Score candidates (cast to float32 for stable RL)
                        scores = self.scorer(h_cog=cog_state.float(), l_curr=l_curr, candidate_layers=candidate_layers)
                        scores = scores.float()
                        # NaN guard: if scorer output is NaN, use uniform distribution
                        if torch.isnan(scores).any():
                            scores = torch.zeros_like(scores)
                        probs = F.softmax(scores / 1.5, dim=-1)
                        probs = probs.clamp(min=1e-8)
                        probs = probs / probs.sum(dim=-1, keepdim=True)  # re-normalize

                        policy_dist = Categorical(probs=probs)
                        action_indices = policy_dist.sample()
                        log_probs = policy_dist.log_prob(action_indices)  # float32

                        l_next_indices = torch.tensor(candidate_layers, device=self.device)[action_indices]

                        # Apply adapter for skipped layers
                        final_group_states = processed_states.clone()
                        skip_mask = (l_next_indices > l_curr + 1)
                        if torch.any(skip_mask):
                            unique_l_skips = torch.unique(l_next_indices[skip_mask])
                            for l_skip_choice in unique_l_skips:
                                sub_group_mask = (l_next_indices == l_skip_choice)
                                sub_group_indices = sub_group_mask.nonzero(as_tuple=True)[0]
                                states_to_adapt = processed_states.index_select(0, sub_group_indices)
                                adapted_states = self.adapter(states_to_adapt.float(), l_curr, l_skip_choice.item())
                                final_group_states.index_copy_(0, sub_group_indices, adapted_states.to(final_group_states.dtype))

                        next_hidden_state.index_copy_(0, group_indices, final_group_states)
                        l_curr_batch.index_copy_(0, group_indices, l_next_indices)
                        step_log_probs.index_copy_(0, group_indices, log_probs)
                        step_kd_loss.index_copy_(0, group_indices, kd_loss.float())

                    student_hidden_state = next_hidden_state
                    log_probs_rollout.append(step_log_probs)
                    kd_loss_rollout.append(step_kd_loss)
                    l_next_rollout.append(l_curr_batch.clone())

                # Execute remaining layers (fix: align training with inference)
                for final_l in range(self.num_layers):
                    needs_exec = (l_curr_batch == final_l)
                    if not torch.any(needs_exec):
                        continue
                    for remaining_l in range(final_l, self.num_layers):
                        exec_indices = needs_exec.nonzero(as_tuple=True)[0]
                        group_states = student_hidden_state.index_select(0, exec_indices)
                        group_pos_emb = (position_embeddings[0].index_select(0, exec_indices),
                                         position_embeddings[1].index_select(0, exec_indices))
                        processed = self._run_single_layer(
                            group_states, remaining_l, causal_mask, group_pos_emb
                        )
                        student_hidden_state.index_copy_(0, exec_indices, processed)
                    l_curr_batch.masked_fill_(needs_exec, self.num_layers)

                # --- Compute losses ---
                logits = self._get_final_logits_and_pred(student_hidden_state, last_token_idx)

                # For BoolQ: extract yes/no logits and compute CE
                binary_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)  # (B, 2)
                loss_ce = F.cross_entropy(binary_logits.float(), labels)
                pred_idx = torch.argmax(binary_logits, dim=-1)
                final_pred_correct = (pred_idx == labels)

                # RL loss
                num_steps = len(log_probs_rollout)
                if num_steps > 0:
                    rewards = torch.zeros(num_steps, batch_size, device=self.device)
                    for t in range(num_steps):
                        for i in range(batch_size):
                            if l_curr_rollout[t][i] < self.num_layers - 1:
                                is_final = (t == num_steps - 1) or (l_next_rollout[t][i] >= self.num_layers - 1)
                                rewards[t, i] = self.reward_fn.compute_reward(
                                    final_pred_correct[i].item(),
                                    l_curr_rollout[t][i].item(),
                                    l_next_rollout[t][i].item(),
                                    is_final, t, num_steps
                                )
                    discounted_rewards = torch.zeros_like(rewards)
                    R = torch.zeros(batch_size, device=self.device)
                    for t in reversed(range(num_steps)):
                        R = rewards[t] + self.config.GAMMA * R
                        discounted_rewards[t] = R
                    # Normalize across the full tensor (more stable than per-sample)
                    mean = discounted_rewards.mean()
                    std = discounted_rewards.std() + 1e-6
                    discounted_rewards = (discounted_rewards - mean) / std
                    discounted_rewards = discounted_rewards.clamp(-5.0, 5.0)  # prevent extreme values
                    log_probs_tensor = torch.stack(log_probs_rollout)
                    policy_loss = (-log_probs_tensor * discounted_rewards).sum(dim=0).mean()
                    total_kd_loss = torch.stack(kd_loss_rollout).sum(dim=0).mean()

                    # NaN guard: if any loss is NaN, skip this step
                    if torch.isnan(policy_loss) or torch.isnan(total_kd_loss):
                        policy_loss = torch.tensor(0.0, device=self.device)
                        total_kd_loss = torch.tensor(0.0, device=self.device)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                    total_kd_loss = torch.tensor(0.0, device=self.device)

                total_loss = (self.config.CE_LOSS_WEIGHT * loss_ce +
                              self.config.RL_LOSS_WEIGHT * policy_loss +
                              self.config.KD_LOSS_WEIGHT * total_kd_loss)

                # NaN guard on total loss: skip backward if NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if self.rank == 0:
                        print(f"[WARN] NaN/Inf loss at step {step}, skipping. "
                              f"CE={loss_ce.item():.4f}, RL={policy_loss.item()}, KD={total_kd_loss.item()}")
                    self.optimizer.zero_grad()
                    continue

                (total_loss / self.config.GRADIENT_ACCUMULATION_STEPS).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # Check for NaN in gradients before stepping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.scorer.parameters()) + list(self.adapter.parameters()),
                        self.config.CLIP_GRAD_NORM
                    )
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if self.rank == 0:
                            print(f"[WARN] NaN/Inf grad_norm at step {step}, skipping optimizer step.")
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                if self.rank == 0 and (step + 1) % self.config.LOG_INTERVAL == 0:
                    pbar.set_postfix({
                        "Loss": f"{total_loss.item():.4f}",
                        "CE": f"{loss_ce.item():.4f}",
                        "RL": f"{policy_loss.item():.8f}",
                        "KD": f"{total_kd_loss.item():.4f}"
                    })

                # TensorBoard + wandb logging
                if self.writer is not None:
                    global_step += 1
                    metrics = {
                        'Loss/total': total_loss.item(),
                        'Loss/CE': loss_ce.item(),
                        'Loss/RL_policy': policy_loss.item(),
                        'Loss/KD': total_kd_loss.item(),
                        'Loss/CE_weighted': self.config.CE_LOSS_WEIGHT * loss_ce.item(),
                        'Loss/RL_weighted': self.config.RL_LOSS_WEIGHT * policy_loss.item(),
                        'Loss/KD_weighted': self.config.KD_LOSS_WEIGHT * total_kd_loss.item(),
                        'Train/accuracy': final_pred_correct.float().mean().item(),
                        'Train/avg_decision_steps': num_steps,
                        'System/gpu_mem_GB': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    }
                    for k, v in metrics.items():
                        self.writer.add_scalar(k, v, global_step)
                    wandb.log(metrics, step=global_step)

                epoch_correct += final_pred_correct.sum().item()
                epoch_total += batch_size

            # Epoch-level logging
            if self.writer is not None:
                epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                self.writer.add_scalar('Epoch/accuracy', epoch_acc, epoch + 1)
                self.writer.flush()
                wandb.log({'Epoch/accuracy': epoch_acc, 'epoch': epoch + 1})

            if self.rank == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, "aster_llama_checkpoint.pt")
        scorer_state_dict = self.scorer.module.state_dict() if self.world_size > 1 else self.scorer.state_dict()
        adapter_state_dict = self.adapter.module.state_dict() if self.world_size > 1 else self.adapter.state_dict()
        checkpoint = {
            'epoch': epoch,
            'scorer_state_dict': scorer_state_dict,
            'adapter_state_dict': adapter_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch + 1} to {checkpoint_path}")

    def train_mmlu(self, train_dataloader, tokenizer, choice_ids, train_sampler=None, start_epoch=0):
        """
        MMLU training: 4-choice (A/B/C/D) instead of binary (Yes/No).
        Reuses the same MDP rollout logic, only differs in loss computation.
        """
        if self.rank == 0:
            print("--- Starting ASTER-MMLU Training ---")

        if start_epoch == 0:
            self.optimizer.zero_grad()
        global_step = start_epoch * len(train_dataloader)

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            if self.rank == 0:
                print(f"\n--- Epoch {epoch + 1}/{self.config.NUM_EPOCHS} ---")
            if self.world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(self.rank != 0))
            epoch_correct, epoch_total = 0, 0

            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size = input_ids.shape[0]
                last_token_idx = self._get_last_token_indices(attention_mask)

                # Teacher forward
                with torch.no_grad():
                    teacher_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                                 output_hidden_states=True, output_attentions=False)
                    teacher_hidden_states = teacher_outputs.hidden_states
                    teacher_attentions = None

                # Student MDP rollout (identical to BoolQ version)
                student_hidden_state = self.model.model.embed_tokens(input_ids)
                position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.model.model.rotary_emb(student_hidden_state, position_ids)
                causal_mask = None

                log_probs_rollout, kd_loss_rollout = [], []
                l_curr_rollout, l_next_rollout = [], []
                l_curr_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                executed_layers_count = 0

                while True:
                    active_mask = l_curr_batch < self.num_layers - 1
                    if not torch.any(active_mask):
                        break
                    executed_layers_count += 1
                    l_curr_rollout.append(l_curr_batch.clone())
                    next_hidden_state = student_hidden_state.clone()
                    step_log_probs = torch.zeros(batch_size, device=self.device)
                    step_kd_loss = torch.zeros(batch_size, device=self.device)
                    unique_l_currs = torch.unique(l_curr_batch[active_mask])

                    for l_c in unique_l_currs:
                        l_curr = l_c.item()
                        group_indices = (l_curr_batch == l_curr).nonzero(as_tuple=True)[0]
                        group_hidden_states = student_hidden_state.index_select(0, group_indices)
                        group_pos_emb = (position_embeddings[0].index_select(0, group_indices),
                                         position_embeddings[1].index_select(0, group_indices))

                        processed_states = self._run_single_layer(
                            group_hidden_states, l_curr, causal_mask, group_pos_emb)

                        teacher_ref_state = teacher_hidden_states[l_curr + 1].detach().index_select(0, group_indices)
                        group_last_idx = last_token_idx.index_select(0, group_indices)
                        kd_loss = self._compute_knowledge_distillation_loss(
                            processed_states.float(), teacher_ref_state.float(), None, group_last_idx)

                        cog_state = self._get_cognitive_token(processed_states, group_last_idx)

                        required_future_steps = self.config.MIN_TOTAL_EXECUTED_LAYERS - executed_layers_count
                        max_allowed_l_next = self.num_layers - required_future_steps \
                            if required_future_steps > 0 else self.num_layers
                        start_candidate = l_curr + 1
                        end_candidate = min(self.num_layers, max_allowed_l_next + 1)
                        candidate_layers = list(range(start_candidate, end_candidate))
                        if not candidate_layers and start_candidate < self.num_layers:
                            candidate_layers = [start_candidate]
                        if not candidate_layers:
                            l_curr_batch.index_fill_(0, group_indices, self.num_layers)
                            next_hidden_state.index_copy_(0, group_indices, processed_states)
                            continue

                        scores = self.scorer(h_cog=cog_state.float(), l_curr=l_curr, candidate_layers=candidate_layers)
                        scores = scores.float()
                        if torch.isnan(scores).any():
                            scores = torch.zeros_like(scores)
                        probs = F.softmax(scores / 1.5, dim=-1)
                        probs = probs.clamp(min=1e-8)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                        policy_dist = Categorical(probs=probs)
                        action_indices = policy_dist.sample()
                        log_probs = policy_dist.log_prob(action_indices)
                        l_next_indices = torch.tensor(candidate_layers, device=self.device)[action_indices]

                        final_group_states = processed_states.clone()
                        skip_mask = (l_next_indices > l_curr + 1)
                        if torch.any(skip_mask):
                            unique_l_skips = torch.unique(l_next_indices[skip_mask])
                            for l_skip_choice in unique_l_skips:
                                sub_mask = (l_next_indices == l_skip_choice)
                                sub_idx = sub_mask.nonzero(as_tuple=True)[0]
                                adapted = self.adapter(
                                    processed_states.index_select(0, sub_idx).float(),
                                    l_curr, l_skip_choice.item()
                                ).to(final_group_states.dtype)
                                final_group_states.index_copy_(0, sub_idx, adapted)

                        next_hidden_state.index_copy_(0, group_indices, final_group_states)
                        l_curr_batch.index_copy_(0, group_indices, l_next_indices)
                        step_log_probs.index_copy_(0, group_indices, log_probs)
                        step_kd_loss.index_copy_(0, group_indices, kd_loss.float())

                    student_hidden_state = next_hidden_state
                    log_probs_rollout.append(step_log_probs)
                    kd_loss_rollout.append(step_kd_loss)
                    l_next_rollout.append(l_curr_batch.clone())

                # Execute remaining layers
                for final_l in range(self.num_layers):
                    needs_exec = (l_curr_batch == final_l)
                    if not torch.any(needs_exec):
                        continue
                    for remaining_l in range(final_l, self.num_layers):
                        exec_indices = needs_exec.nonzero(as_tuple=True)[0]
                        group_states = student_hidden_state.index_select(0, exec_indices)
                        group_pos_emb = (position_embeddings[0].index_select(0, exec_indices),
                                         position_embeddings[1].index_select(0, exec_indices))
                        processed = self._run_single_layer(
                            group_states, remaining_l, causal_mask, group_pos_emb)
                        student_hidden_state.index_copy_(0, exec_indices, processed)
                    l_curr_batch.masked_fill_(needs_exec, self.num_layers)

                # MMLU: 4-choice logits
                logits = self._get_final_logits_and_pred(student_hidden_state, last_token_idx)
                choice_logits = torch.stack([logits[:, cid] for cid in choice_ids], dim=-1)  # (B, 4)
                loss_ce = F.cross_entropy(choice_logits.float(), labels)
                pred_idx = torch.argmax(choice_logits, dim=-1)
                final_pred_correct = (pred_idx == labels)

                # RL loss
                num_steps = len(log_probs_rollout)
                if num_steps > 0:
                    rewards = torch.zeros(num_steps, batch_size, device=self.device)
                    for t in range(num_steps):
                        for i in range(batch_size):
                            if l_curr_rollout[t][i] < self.num_layers - 1:
                                is_final = (t == num_steps - 1) or (l_next_rollout[t][i] >= self.num_layers - 1)
                                rewards[t, i] = self.reward_fn.compute_reward(
                                    final_pred_correct[i].item(),
                                    l_curr_rollout[t][i].item(),
                                    l_next_rollout[t][i].item(),
                                    is_final, t, num_steps)
                    discounted_rewards = torch.zeros_like(rewards)
                    R = torch.zeros(batch_size, device=self.device)
                    for t in reversed(range(num_steps)):
                        R = rewards[t] + self.config.GAMMA * R
                        discounted_rewards[t] = R
                    mean = discounted_rewards.mean()
                    std = discounted_rewards.std() + 1e-6
                    discounted_rewards = (discounted_rewards - mean) / std
                    discounted_rewards = discounted_rewards.clamp(-5.0, 5.0)
                    log_probs_tensor = torch.stack(log_probs_rollout)
                    policy_loss = (-log_probs_tensor * discounted_rewards).sum(dim=0).mean()
                    total_kd_loss = torch.stack(kd_loss_rollout).sum(dim=0).mean()
                    if torch.isnan(policy_loss) or torch.isnan(total_kd_loss):
                        policy_loss = torch.tensor(0.0, device=self.device)
                        total_kd_loss = torch.tensor(0.0, device=self.device)
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                    total_kd_loss = torch.tensor(0.0, device=self.device)

                total_loss = (self.config.CE_LOSS_WEIGHT * loss_ce +
                              self.config.RL_LOSS_WEIGHT * policy_loss +
                              self.config.KD_LOSS_WEIGHT * total_kd_loss)

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if self.rank == 0:
                        print(f"[WARN] NaN/Inf loss at step {step}, skipping.")
                    self.optimizer.zero_grad()
                    continue

                (total_loss / self.config.GRADIENT_ACCUMULATION_STEPS).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.scorer.parameters()) + list(self.adapter.parameters()),
                        self.config.CLIP_GRAD_NORM)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                if self.rank == 0 and (step + 1) % self.config.LOG_INTERVAL == 0:
                    pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "CE": f"{loss_ce.item():.4f}",
                                      "RL": f"{policy_loss.item():.8f}", "KD": f"{total_kd_loss.item():.4f}"})

                if self.writer is not None:
                    global_step += 1
                    metrics = {
                        'Loss/total': total_loss.item(),
                        'Loss/CE': loss_ce.item(),
                        'Loss/RL_policy': policy_loss.item(),
                        'Loss/KD': total_kd_loss.item(),
                        'Train/accuracy': final_pred_correct.float().mean().item(),
                        'Train/avg_decision_steps': num_steps,
                    }
                    for k, v in metrics.items():
                        self.writer.add_scalar(k, v, global_step)
                    try:
                        wandb.log(metrics, step=global_step)
                    except Exception:
                        pass

                epoch_correct += final_pred_correct.sum().item()
                epoch_total += batch_size

            if self.writer is not None:
                epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                self.writer.add_scalar('Epoch/accuracy', epoch_acc, epoch + 1)
                self.writer.flush()

            if self.rank == 0:
                self.save_checkpoint(epoch)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.rank == 0:
            wandb.finish()

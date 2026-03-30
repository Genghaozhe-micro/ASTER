# ASTER-BERT 代码解读文档

## 1. 论文概要

**ASTER** (*Adaptive Dynamic Layer-Skipping for Efficient Transformer Inference via Markov Decision Process*) 发表于 ACM MM 2025（The 33rd ACM International Conference on Multimedia, Dublin, Ireland）。

### 1.1 研究动机

Transformer 模型在 NLP 和 CV 任务上表现优异，但层数增加带来了巨大的计算开销和推理延迟。研究发现，**并非所有层对每个输入都同等重要**——某些输入在较浅层就已获得足够信息，继续计算反而可能引入噪声或降低性能（论文 Figure 1 展示了 LLAMA3.1-8B 在中间层就已预测正确，但最终层反而输出了错误答案）。

### 1.2 核心思想

将 Transformer 的**动态层跳过**决策建模为**马尔可夫决策过程（MDP）**：

| MDP 要素 | 对应含义 |
|---|---|
| **状态 (S)** | 当前层的隐藏状态（特别是 Cognitive Token — 即 `[CLS]` token 的表示）+ 当前层索引 |
| **动作 (A)** | 选择跳到哪一层（可以跳过中间多层） |
| **策略 (π)** | 由 Scorer 网络决定跳跃目标层 |
| **奖励 (R)** | TIDR（Time-aware Importance-weighted Dynamic Reward），综合效率奖励、跳步惩罚和任务奖励 |
| **转移 (P)** | 执行选定层后得到新的隐藏状态，进入下一个决策步 |

### 1.3 关键技术贡献

1. **MDP 建模的自适应层跳过**：根据每个输入样本的中间状态动态决定计算路径
2. **Cognitive Token**：复用 `[CLS]` token（BERT）或最后一个 token（LLaMA）作为认知令牌，用于捕获当前计算状态
3. **Dynamic Adapter**：当层被跳过时，通过适配器网络补偿信息损失
4. **TIDR 奖励机制**：解决层跳过决策中的信用分配问题
5. **Cognition-Based 知识蒸馏**：针对不同架构（Encoder / Decoder）设计不同的蒸馏策略

### 1.4 实验结果（DistilBERT on SST-2）

| 加速比 | Baseline | Early-exit | Random-skip | Skipdecode | Shortgpt | **ASTER (Ours)** |
|---|---|---|---|---|---|---|
| 1.0× | 92.70% | — | — | — | — | — |
| 1.2× | — | 90.48% | 87.84% | 88.07% | 88.53% | **88.30%** |
| 1.5× | — | 73.97% | 80.05% | 77.06% | 77.41% | **84.75%** |
| 3.0× | — | 49.20% | 71.56% | 69.61% | 69.61% | **81.65%** |

在较高加速比下（1.5× 和 3.0×），ASTER 显著优于所有基线方法。

---

## 2. 运行环境

### 2.1 环境配置

| 项目 | 版本/值 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| transformers | 5.4.0 |
| datasets | 4.8.4 |
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA Driver | 570.195.03 (CUDA 12.8) |

### 2.2 环境创建与激活

```bash
# 创建 conda 环境
conda create -n aster python=3.10 -y

# 激活环境
eval "$(conda shell.bash hook)" && conda activate aster

# 安装依赖
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets tqdm numpy
```

### 2.3 验证环境

```bash
cd aster-bert
python -c "
import config_bert as config
from components import ScoringModel, DynamicAdapter
from reward import TIDR_V2
print('All modules imported successfully.')
print(f'Model: {config.MODEL_ID}')
print(f'Dataset: {config.DATASET_NAME}/{config.DATASET_CONFIG}')
print(f'Device: {config.DEVICE}')
"
```

---

## 3. 项目结构

```
aster-bert/
├── config_bert.py              # 全局配置（超参数、模型、数据集）
├── components.py               # ASTER 核心组件（ScoringModel + DynamicAdapter）
├── reward.py                   # TIDR 奖励函数
├── model_utils_bert.py         # 模型/分词器加载工具
├── bert_trainer.py             # 训练核心逻辑（MDP rollout + 3项损失）
├── train_bert.py               # 单卡训练入口
├── train_bert_distributed.py   # 多卡分布式训练入口
├── evaluate_bert_aster.py      # ASTER 贪心推理评估
└── baseline.py                 # 完整模型基线评估
```

### 模块依赖关系

```
config_bert.py ◄──────────────────── 被所有模块引用
       │
model_utils_bert.py ◄─── 加载预训练 DistilBERT 模型
       │
components.py ◄───────── ScoringModel (策略网络) + DynamicAdapter (跳层补偿)
       │
reward.py ◄────────────── TIDR 奖励计算
       │
bert_trainer.py ◄──────── 训练核心：组合以上所有模块
       │
       ├── train_bert.py ────────────── 单 GPU 训练入口
       ├── train_bert_distributed.py ── 多 GPU DDP 训练入口
       ├── evaluate_bert_aster.py ───── ASTER 评估
       └── baseline.py ─────────────── 基线评估
```

---

## 4. 各模块详细解读

### 4.1 `config_bert.py` — 全局配置

定义所有超参数和配置常量，是整个项目的"控制中心"。

```python
# 模型配置
MODEL_ID = 'distilbert-base-uncased-finetuned-sst-2-english'  # 6层 DistilBERT，在 SST-2 上微调

# 数据集配置
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"          # 二分类情感分析

# 训练超参
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 128

# ASTER 组件维度
ADAPTER_BOTTLENECK_DIM = 128     # Adapter 瓶颈维度
SCORER_HIDDEN_DIM = 256          # Scorer MLP 隐藏维度

# 损失权重（对应论文公式 11 的 λ）
CE_LOSS_WEIGHT = 10              # λ_TASK：任务损失权重
RL_LOSS_WEIGHT = 0.5             # λ_RL：策略梯度损失权重
KD_LOSS_WEIGHT = 1.0             # λ_KD：知识蒸馏损失权重

# 强化学习超参（对应论文公式 13）
GAMMA = 0.99                     # 折扣因子 γ
W_EFFICIENCY = 0.01              # λ_eff：效率奖励权重
W_TASK = 1.0                     # λ_task：任务奖励权重
SKIP_PENALTY_WEIGHT = 0.1        # λ_skip：跳步惩罚权重

# 最小执行层数约束
MIN_TOTAL_EXECUTED_LAYERS = 2    # 防止过于激进的跳层
```

**论文对应**：公式 (11) 中的 $\lambda_{TASK}$、$\lambda_{RL}$、$\lambda_{KD}$ 及公式 (13) 中的 $\lambda_{eff}$、$\lambda_{skip}$、$\lambda_{task}$。

---

### 4.2 `components.py` — ASTER 核心组件

包含论文中两个关键新增模块。

#### 4.2.1 `DynamicAdapter`（对应论文 §3.4）

**作用**：当层被跳过时，补偿跳过层导致的信息丢失。

**论文描述**：

> Dynamic Feature Representations are introduced to compensate for the missing computations when layers are skipped.

**结构**：

```
输入 hidden_state (B × N × D)
       │
       ├── Layer Embedding: embed(l_curr), embed(l_next)
       │       │
       │       └── skip_signal = embed(l_next) - embed(l_curr)  ← 跳跃信号
       │
       ├── conditioned_state = hidden_state + skip_signal
       │
       ├── Bottleneck MLP: Linear(D→128) → GELU → Linear(128→D)
       │
       └── LayerNorm(residual + mlp_output)  ← 残差连接
```

**代码解读**：

```python
class DynamicAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, num_layers, dtype):
        # 层嵌入：为每一层学习一个向量表示
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)
        # 瓶颈 MLP：先降维再升维，学习跳跃的非线性映射
        self.adapter_network = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),  # 768 → 128
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim)   # 128 → 768
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_state, l_curr, l_next):
        residual = hidden_state
        # 计算跳跃信号：目标层嵌入 - 当前层嵌入
        skip_signal = self.layer_embedding(l_next) - self.layer_embedding(l_curr)
        # 将跳跃信号注入隐藏状态
        conditioned_state = hidden_state + skip_signal
        # 通过瓶颈网络转换
        adapter_output = self.adapter_network(conditioned_state)
        # 残差连接 + LayerNorm
        return self.layer_norm(residual + adapter_output)
```

**设计思想**：通过学习层间的嵌入差值（skip_signal），让 Adapter 知道"跳过了什么"，从而生成补偿特征。瓶颈结构保证参数效率。

---

#### 4.2.2 `ScoringModel`（对应论文 §3.1 策略网络 $\pi_\theta$）

**作用**：作为 MDP 的策略网络，评估每个候选目标层的得分，决定跳到哪一层。

**论文对应**：公式 (2) $l_{t+1} \sim \pi_\theta(l_{t+1} | l_t, h_t)$

**结构**：

```
输入: h_cls (B × D)     ← [CLS] token 的隐藏状态（Cognitive Token）
      l_curr             ← 当前层索引
      candidate_layers   ← 候选目标层列表

对每个候选层 l_cand:
    embed_curr = LayerEmbedding(l_curr)      ← D/4 维
    embed_cand = LayerEmbedding(l_cand)      ← D/4 维
    input = [h_cls, embed_curr, embed_cand]  ← 拼接后 D + D/4 + D/4 = 1.5D
    score = MLP(input)                       ← 标量得分

输出: scores (B × num_candidates)
```

**代码解读**：

```python
class ScoringModel(nn.Module):
    def __init__(self, hidden_dim, scorer_hidden_dim, num_layers, dtype):
        embedding_dim = hidden_dim // 4  # 768//4 = 192
        self.layer_embedding = nn.Embedding(num_layers, embedding_dim)
        self.scorer_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, scorer_hidden_dim),  # 768+384=1152 → 256
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1)  # 256 → 1
        )

    def forward(self, h_cls, l_curr, candidate_layers):
        scores = []
        for l_cand in candidate_layers:
            # 拼接: [CLS]表示 + 当前层嵌入 + 候选层嵌入
            mlp_input = torch.cat([h_cls, embed(l_curr), embed(l_cand)], dim=-1)
            score = self.scorer_mlp(mlp_input)
            scores.append(score)
        return torch.cat(scores, dim=-1)  # (B, num_candidates)
```

**设计思想**：利用 Cognitive Token（`[CLS]`）的当前表示作为"计算状态"，结合层位置信息，让网络学会判断"从当前位置跳到候选位置是否合适"。

---

### 4.3 `reward.py` — TIDR 奖励函数

**作用**：实现论文 §3.5 的 **Time-aware Importance-weighted Dynamic Reward (TIDR)**，为强化学习策略提供奖励信号。

**论文公式 (13)**：

$$r_t = \lambda_{eff} \cdot R_{eff} - \lambda_{skip} \cdot P_{skip} + \lambda_{task} \cdot \omega_t \cdot R_{task} \cdot \mathbb{I}(t = T-1)$$

**三项奖励分量**：

| 分量 | 公式 | 含义 |
|---|---|---|
| 效率奖励 $R_{eff}$ | $\log(1 + l_{next} - l_{curr})$ | 鼓励更大步跳跃（密集奖励） |
| 跳步惩罚 $P_{skip}$ | $\left(\frac{l_{next} - l_{curr}}{L}\right)^2$ | 防止过大跳跃损害性能（二次惩罚） |
| 任务奖励 $R_{task}$ | $+1$ (正确) / $-1$ (错误) | 仅在最后一步给出（稀疏奖励） |

**时间感知权重**：

$$\omega_t = \frac{T - t}{T} \cdot \sigma\left(\frac{l_{next} - l_{curr}}{L}\right)$$

- **时间衰减** $\frac{T-t}{T}$：早期决策对最终结果影响较小
- **跳跃重要性** $\sigma(\cdot)$：跳跃越大，权重越高

**代码解读**：

```python
class TIDR_V2:
    def compute_reward(self, final_prediction_correct, l_curr, l_next,
                       is_final_step, step_t, total_steps):
        # 时间感知权重 ω_t
        time_decay = (total_steps - step_t) / total_steps
        skip_weight = sigmoid((l_next - l_curr) / L)
        omega_t = time_decay * skip_weight

        # 任务奖励（仅最后一步）
        task_reward = (1.0 if correct else -1.0) if is_final_step else 0.0

        # 效率奖励（每步都有）
        efficiency_reward = log(1.001 + l_next - l_curr)

        # 跳步惩罚（防止过大跳跃）
        skip_penalty = ((l_next - l_curr) / L) ** 2

        # 总奖励
        total = w_task * task_reward * omega_t
               + w_efficiency * efficiency_reward
               - skip_penalty_weight * skip_penalty
        return total
```

**设计思想**：通过密集的效率奖励鼓励跳层、二次惩罚防止激进跳跃、稀疏任务奖励保证最终预测质量，三者相互博弈达到动态平衡。

---

### 4.4 `model_utils_bert.py` — 模型加载工具

**作用**：从 HuggingFace Hub（通过 `hf-mirror.com` 镜像）加载预训练的 DistilBERT 模型和 Tokenizer。

```python
def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased-finetuned-sst-2-english',
        output_hidden_states=True,  # 关键：需要输出每层隐藏状态
        num_labels=2                # SST-2 二分类
    )
    model.eval()  # 冻结，作为教师模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer
```

**双重角色**：
- **教师模型**：冻结参数，完整计算所有层，产出 `teacher_hidden_states` 供知识蒸馏
- **架构提供者**：提供 embedding 层和各 transformer 层给学生路径使用

---

### 4.5 `bert_trainer.py` — 核心训练逻辑

**作用**：`ASTERTrainerBERT` 类实现论文的完整训练流程，是整个项目最核心的文件。

#### 4.5.1 训练流程总览

```
对每个 batch:
    ┌─────────────────────────────────────────┐
    │ 1. 教师前向传播（冻结，完整执行所有层）  │
    │    → teacher_hidden_states              │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │ 2. 学生动态路径推理（MDP Rollout）       │
    │    while l_curr < num_layers - 1:       │
    │      a. 执行当前层                       │
    │      b. 提取 [CLS] 作为 Cognitive Token │
    │      c. Scorer 为候选层打分              │
    │      d. 采样动作（目标层）               │
    │      e. 若跳层 → Adapter 补偿            │
    │      f. 计算 KD loss                    │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │ 3. 三项损失计算                          │
    │    L_total = λ_CE·L_CE                  │
    │           + λ_RL·L_RL                   │
    │           + λ_KD·L_KD                   │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │ 4. 反向传播 + 梯度更新                   │
    │    仅更新 Scorer 和 Adapter 参数         │
    └─────────────────────────────────────────┘
```

#### 4.5.2 MDP Rollout 详解

```python
# 学生从 embedding 输出开始
student_hidden_state = model.distilbert.embeddings(input_ids)

while l_curr < num_layers - 1:
    # (a) 执行当前层
    processed = transformer.layer[l_curr](student_hidden_state, attn_mask)

    # (b) 提取 Cognitive Token
    cls_state = processed[:, 0, :]  # [CLS] token

    # (c) Scorer 打分
    scores = scorer(h_cls=cls_state, l_curr=l_curr, candidate_layers=[...])
    probs = softmax(scores / temperature)

    # (d) 采样动作
    action = Categorical(probs).sample()  # REINFORCE 采样
    l_next = candidate_layers[action]

    # (e) 跳层补偿
    if l_next > l_curr + 1:
        student_hidden_state = adapter(processed, l_curr, l_next)

    # (f) 知识蒸馏 loss
    kd_loss = KD(student_state=processed, teacher_state=teacher_hidden_states[l_curr+1])
```

#### 4.5.3 知识蒸馏损失（对应论文 §3.6 公式 14-15）

```python
def _compute_knowledge_distillation_loss(self, student_state, teacher_state):
    # 拆分 [CLS] token 和普通 token
    student_cls, student_tokens = student_state[:, 0], student_state[:, 1:]
    teacher_cls, teacher_tokens = teacher_state[:, 0], teacher_state[:, 1:]

    # 目标1: [CLS] token 的 MSE 对齐（λ_cls 部分）
    loss_cls = MSE(student_cls, teacher_cls)

    # 目标2: 加权 token-level MSE（λ_token 部分）
    # 重要性权重 = σ(T·CLS / τ) × σ(S·CLS / τ)
    # → 只有 teacher 和 student 都认为与 [CLS] 相关的 token 才有高权重
    I_teacher = sigmoid(teacher_tokens @ teacher_cls / τ)
    I_student = sigmoid(student_tokens @ student_cls / τ)
    importance = I_teacher * I_student  # 双向一致性

    loss_tokens = (importance * MSE(student_tokens, teacher_tokens)).mean()

    return loss_cls + loss_tokens
```

**对应公式 (15)**：

$$\mathcal{I}_{BERT}(H_S, H_T, i) = \begin{cases} \lambda_{cls} & \text{if } i = 0 \\ \lambda_{token} \cdot \sigma\!\left(\frac{H_{T,i} \cdot H_{T,0}}{\tau}\right) \cdot \sigma\!\left(\frac{H_{S,i} \cdot H_{S,0}}{\tau}\right) & \text{if } i > 0 \end{cases}$$

#### 4.5.4 三项损失组合（对应论文公式 11）

```python
# L_TASK: 交叉熵分类损失
loss_ce = CrossEntropy(final_logits, labels)

# L_RL: REINFORCE 策略梯度损失（公式 12）
# G_t = Σ γ^(k-t) · r_k（折扣累积奖励）
policy_loss = -Σ(log_prob * G_t)

# L_KD: 知识蒸馏损失（公式 14-15）
total_kd_loss = Σ(kd_loss_per_step)

# 加权组合
total_loss = 10 * loss_ce + 0.5 * policy_loss + 1.0 * total_kd_loss
```

**关键设计**：只有 Scorer 和 Adapter 的参数参与梯度更新，基模型完全冻结。

---

### 4.6 `train_bert.py` — 单卡训练入口

**作用**：单 GPU 训练脚本，串联所有模块完成训练。

**流程**：

```
1. set_seed(42)                    ← 设置随机种子
2. load_model_and_tokenizer()      ← 加载 DistilBERT
3. load_dataset("glue", "sst2")    ← 加载 SST-2 数据集
4. 初始化 ScoringModel + DynamicAdapter
5. 初始化 AdamW 优化器
6. [可选] --resume 恢复 checkpoint
7. ASTERTrainerBERT.train()        ← 开始训练
```

**运行方式**：

```bash
# 从头训练
python train_bert.py

# 恢复训练
python train_bert.py --resume
```

---

### 4.7 `train_bert_distributed.py` — 多卡分布式训练入口

**作用**：使用 PyTorch DDP (DistributedDataParallel) 的多 GPU 训练脚本。

**与单卡版本的主要区别**：

| 特性 | 单卡 (`train_bert.py`) | 多卡 (`train_bert_distributed.py`) |
|---|---|---|
| 进程模型 | 单进程 | `mp.spawn` 多进程 |
| 数据分割 | `DataLoader(shuffle=True)` | `DistributedSampler` |
| 模型包装 | 原始模型 | `DDP(scorer)`, `DDP(adapter)` |
| 学习率 | `5e-5` | `5e-5 × world_size`（线性缩放） |
| 最低要求 | 1 GPU | 2+ GPU |

**运行方式**：

```bash
# 自动检测 GPU 数量并启动
python train_bert_distributed.py

# 恢复训练
python train_bert_distributed.py --resume
```

---

### 4.8 `evaluate_bert_aster.py` — ASTER 评估脚本

**作用**：加载训练好的 Scorer/Adapter checkpoint，在验证集上使用**贪心推理**评估。

**与训练时的区别**：
- 训练时：Categorical 采样（探索）
- 评估时：**argmax 贪心**（利用）

**推理流程**（`predict_greedy` 函数）：

```
embedding → 循环:
    执行当前层 → Scorer 打分 → argmax选最高分的目标层
    → 若跳层则 Adapter 补偿 → 前进到目标层
→ 执行剩余层 → 分类头输出预测
```

**输出指标**：

- Top-1 Accuracy：分类准确率
- Average Executed Layers：平均执行层数
- Computational Savings：计算节省百分比
- Effective Speedup：等效加速比（如 6 层只执行 4 层 → 1.5×）

**运行方式**：

```bash
# 使用默认 checkpoint 路径
python evaluate_bert_aster.py

# 指定 checkpoint
python evaluate_bert_aster.py --checkpoint_path ./checkpoints/aster_bert_checkpoint.pt
```

---

### 4.9 `baseline.py` — 基线评估脚本

**作用**：对原始完整模型（不做任何跳层）在验证集上评估，获取 baseline 准确率。

```python
# 核心逻辑：标准的完整前向传播
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
predictions = torch.argmax(outputs.logits, dim=-1)
accuracy = (predictions == labels).sum() / total
```

**用途**：作为 ASTER 方法的对比基准（论文中 DistilBERT on SST-2 baseline = 92.70%）。

**运行方式**：

```bash
python baseline.py
```

---

## 5. 论文公式与代码对应关系

| 论文公式 | 含义 | 代码位置 |
|---|---|---|
| 公式 (2) $l_{t+1} \sim \pi_\theta(l_{t+1} \mid l_t, h_t)$ | 策略网络选择下一层 | `components.py: ScoringModel.forward()` |
| 公式 (4-5) Cognitive Token | 认知令牌提取 | `bert_trainer.py: cls_state = processed[:, 0, :]` |
| 公式 (11) $\mathcal{L}_{total}$ | 总损失 = CE + RL + KD | `bert_trainer.py: total_loss = CE + RL + KD` |
| 公式 (12) $\mathcal{L}_{RL}$ (REINFORCE) | 策略梯度损失 | `bert_trainer.py: policy_loss = (-log_probs * G_t).mean()` |
| 公式 (13) $r_t$ (TIDR) | 时间感知奖励 | `reward.py: TIDR_V2.compute_reward()` |
| 公式 (14-15) $\mathcal{L}_{KD}$ | BERT 专用知识蒸馏 | `bert_trainer.py: _compute_knowledge_distillation_loss()` |
| §3.4 Dynamic Adapter | 跳层信息补偿 | `components.py: DynamicAdapter.forward()` |
| Algorithm 1 | 动态跳层推理 | `evaluate_bert_aster.py: predict_greedy()` |

---

## 6. 快速上手

```bash
# 1. 激活环境
eval "$(conda shell.bash hook)" && conda activate aster
cd aster-bert

# 2. 运行基线评估（获取完整模型准确率）
python baseline.py

# 3. 训练 ASTER
python train_bert.py

# 4. 评估 ASTER
python evaluate_bert_aster.py
```

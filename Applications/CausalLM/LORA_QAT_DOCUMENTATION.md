# LoRA QAT with Quantized Base Model — Implementation Documentation

This document explains the complete pipeline for fine-tuning a **quantized** LLM
(Qwen3-0.6B stored in Q4_0 format) using **LoRA adapters** trained with
**Quantization-Aware Training (QAT)**. It is written to be understandable without
prior knowledge of the nntrainer internals.

---

## Table of Contents

1. [Background — Why Quantize at All?](#1-background)
2. [Q4_0 GGML Quantization Format (PTQ)](#2-q4_0-ggml-quantization-format)
3. [LoRA — Low-Rank Adaptation](#3-lora--low-rank-adaptation)
4. [The Problem: LoRA on a Quantized Base](#4-the-problem-lora-on-a-quantized-base)
5. [QAT — Quantization-Aware Training on LoRA Adapters](#5-qat--quantization-aware-training-on-lora-adapters)
6. [Per-Block EMA Scale Tracking](#6-per-block-ema-scale-tracking)
7. [Force-Feeding EMA Scales at Save Time](#7-force-feeding-ema-scales-at-save-time)
8. [W4A8 GEMM — Inference Kernel](#8-w4a8-gemm--inference-kernel)
9. [Training Pipeline](#9-training-pipeline)
10. [Inference Pipeline](#10-inference-pipeline)
11. [LAMP Personalization](#11-lamp-personalization)
12. [Memory Breakdown](#12-memory-breakdown)
13. [Data Flow Diagrams](#13-data-flow-diagrams)
14. [File Reference](#14-file-reference)

---

## 1. Background

A large language model like Qwen3-0.6B has ~620 million parameters stored as 32-bit
floats (FP32). That is ~2.4 GB just for weights. To run on mobile devices or edge
hardware with limited RAM, the weights are *quantized* — compressed to 4-bit integers
— reducing size to ~0.6 GB.

**Post-Training Quantization (PTQ)** converts a pre-trained FP32 model to a smaller
format after training is complete. The model is never retrained; the weights are simply
compressed. This is a one-time offline step.

The trade-off: quantization introduces a small but nonzero *quantization error*. For
most tasks the accuracy loss is negligible, but for fine-tuning (adapting the model
to new data), this error matters more.

---

## 2. Q4_0 GGML Quantization Format

### What is Q4_0?

Q4_0 is a block-wise symmetric 4-bit quantization scheme originally from the GGML
library (used by llama.cpp). Every 32 consecutive weight values (a *block*) share one
FP16 scale factor `d`. Each value is stored as a 4-bit integer in range [0, 15].

```
block_q4_0 (one block = 32 elements, 18 bytes total):
┌──────────────┬──────────────────────────────────┐
│  d (FP16)    │  qs[16]  (16 bytes of nibbles)   │
│  2 bytes     │  32 × 4-bit values packed 2/byte │
└──────────────┴──────────────────────────────────┘
```

**Encoding a block:**
```
d = max_abs(block_values) / 7      (scale factor)
q_i = round(value_i / d) + 8      (offset by 8 so range is [0,15] not [-7,7])
```

**Decoding a block:**
```
value_i = (q_i - 8) × d
```

The `+8` / `-8` offset is the key: stored nibbles are *unsigned* [0,15], but logically
they represent signed values [-8,7] centered at zero.

### PTQ: Quantizing the Base Model

The utility `nntr_quantize` walks every FC (fully-connected) layer in the FP32 base
model and calls `quantize_q4_0(fp32_data, …)` on each weight matrix. The result is
stored in the binary weight file `nntr_qwen3_0.6b_q40_embdfp32.bin`.

The embedding layer and LM head are kept in FP32 (richer vocabulary representation
matters more for output quality). Only the FC layers (attention projections, FFN
gates) are Q4_0.

### Repacked Format (block_q4_0x4)

For performance, nntrainer does **not** store Q4_0 in the natural GGML order. After
quantization, a second step `repack_q4_0` interleaves 4 rows at once into a tiled
layout (called `block_q4_0x4`). Additionally, every nibble is XOR'd with `0x88` to
convert the unsigned [0,15] representation into signed [-8,7] representation directly
in the bytes. This pre-processing means the W4A8 GEMM kernel can skip the subtract-8
step at runtime.

```
Natural Q4_0:    nibble in [0,15],  scale = d (FP16)
Repacked Q4_0:   nibble XOR 0x8 ∈ [-8,7],  same scale d
```

At inference load time: `load_weight()` calls `quantize_q4_0` then `repack_q4_0`.
The stored binary file always contains the repacked format.

At dequantization time: `GgmlQuantizer::dequantize` first calls `unpack_q4_0`
(reverses the XOR and tiling) then `dequantize_row_q4_0` (applies scale factors to
get FP32 back).

---

## 3. LoRA — Low-Rank Adaptation

### The Core Idea

Fine-tuning an entire large model requires gradients and optimizer states for every
parameter — too expensive in memory and compute for edge devices.

LoRA (Hu et al. 2021) observes that the weight *update* needed for fine-tuning has
low intrinsic rank. Instead of updating W directly, LoRA adds two small matrices:

```
W_effective = W_frozen + scaling × B × A

where:
  W ∈ R^{N×K}   — original frozen weight
  A ∈ R^{r×K}   — loraA (trained), r ≪ min(N,K)
  B ∈ R^{N×r}   — loraB (trained)
  scaling = alpha / rank
```

Only A and B are updated during training. W is never modified. The memory cost is
`r×K + N×r` instead of `N×K` — for rank=32 on a 1024×1024 matrix, that is 65,536
parameters instead of 1,048,576 — a **16× reduction**.

### Initialization

- **loraA**: initialized with LeCun normal (small random values)
- **loraB**: initialized to zero

At step 0: `B×A = 0`, so `W_effective = W_frozen`. The model starts identical to the
base model and gradually shifts as loraB accumulates gradient.

### Forward Pass (training, full sequence)

```
hidden = input × W^T            (base model path, W is Q4_0 frozen)
tmp    = input × loraA^T        (r-dimensional bottleneck)
out    = tmp   × loraB^T        (back to N-dimensional output)
hidden += scaling × out         (add LoRA residual)
```

### Backward Pass

Only loraA and loraB receive gradients. W has `trainable=false` so nntrainer skips
its gradient computation entirely. This is the key memory saving during training.

### LoRA Targets (Qwen3-0.6B)

LoRA is applied to these 7 projection types per transformer layer (28 layers total =
196 LoRA pairs):

| Target      | Matrix shape  | Description              |
|-------------|---------------|--------------------------|
| `wq`        | 1024 × 1024   | Query projection         |
| `wk`        | 512 × 1024    | Key projection (GQA)     |
| `wv`        | 512 × 1024    | Value projection (GQA)   |
| `wo`        | 1024 × 1024   | Output projection        |
| `ffn_up`    | 2816 × 1024   | FFN up gate              |
| `ffn_gate`  | 2816 × 1024   | FFN gate (SwiGLU)        |
| `ffn_down`  | 1024 × 2816   | FFN down projection      |

---

## 4. The Problem: LoRA on a Quantized Base

Standard LoRA assumes the base model is in FP32. When the base is Q4_0:

1. **Forward pass**: The Q4_0 weight contributes quantization noise to every
   activation. loraA/loraB are trained to *compensate* for this noise.
2. **Save and reload**: If loraA/loraB are saved as FP32 but the base stays Q4_0,
   inference is slightly mismatched — the Q4_0 base has a fixed noise floor that
   the FP32 adapters need to overcome.
3. **Quantizing the adapters**: If we then quantize loraA/loraB to Q4_0 to save
   memory, the adapters themselves introduce additional quantization error on top of
   the base model's error. Without training the adapters to be aware of their own
   quantization, this double-error degrades output quality.

---

## 5. QAT — Quantization-Aware Training on LoRA Adapters

**Quantization-Aware Training (QAT)** solves the adapter quantization problem by
*simulating* quantization during training. The adapters learn weights that are robust
to being stored in Q4_0.

### Fake Quantization

Instead of truly storing loraA/loraB as 4-bit integers (which would destroy
gradients), QAT uses **fake quantization**: a differentiable operation that
*mimics* Q4_0 rounding in the forward pass while passing gradients through unchanged
in the backward pass (Straight-Through Estimator).

```
Forward:   a_fq = dequantize(quantize(loraA))
           ≈ loraA rounded to Q4_0 grid, same dtype (FP32)

Backward:  d(loss)/d(loraA) = d(loss)/d(a_fq)   ← gradient passes through unchanged
```

The "fake-quantized" tensor `a_fq` has the *values* that loraA would have after a
real Q4_0 round-trip (quantize then dequantize), but is still stored as FP32 so
gradients can flow.

### Why This Helps

The model trains to minimize loss *with* the quantization noise included. By the end
of training, loraA and loraB have values that are already "rounded" to the Q4_0 grid
— saving them as Q4_0 at the end introduces almost zero additional error.

### Implementation (`fakeQuantizeQ4_0` in `fc_layer.cpp`)

```cpp
Tensor fakeQuantizeQ4_0(const Tensor &x, std::vector<float> &block_d, bool training):
  for each block of 32 elements in x:
    if first call (bootstrap): d = max_abs(block) / 7
    else if training: d = (1-momentum) * d_ema + momentum * (max_abs(block) / 7)
    else (validation): d = d_ema  (frozen, no update)
    
    quantize: q_i = clamp(round(x_i / d), -8, 7)
    dequantize: x_fq_i = q_i * d
  return x_fq  (FP32, same shape as x)
```

The Q4_0 operation is applied in the **N×K** layout of the weight matrix, with block
boundaries aligned along K (the input dimension), matching the block boundaries used
by the GEMM kernel at inference time.

---

## 6. Per-Block EMA Scale Tracking

### The Challenge

Standard Q4_0 quantization computes the scale factor `d` for each block from
`max_abs(block)`. This is fine for static inference, but during training the
weight values change every step — naively recomputing from max_abs gives noisy,
unstable scale estimates.

### Exponential Moving Average (EMA)

Instead of using the instantaneous max_abs, nntrainer maintains a running EMA of
the scale for each block:

```
d_new = (1 - momentum) × d_old  +  momentum × (max_abs(block) / 7)
momentum = 0.1   (decays fast enough to track changes, slow enough to be stable)
```

This is analogous to BatchNorm's running statistics — the EMA scale is a *calibrated*
estimate of the block's dynamic range accumulated over all training steps.

### Layout: N×K vs K×N

A weight matrix is stored in K×N layout (input dimension K, output dimension N) in
nntrainer. But the Q4_0 GEMM processes blocks along the K dimension (one block = 32
input elements → one partial output). The EMA scales are therefore tracked in the
**N×K** transposed layout so that block index `nk = n*K_blocks + k_block` directly
maps to the GEMM tile boundary.

The mapping between storage index and EMA index:
```
storage index (K×N layout): i = k * N + n
EMA index (N×K layout):     j = n * K_blocks + k_block
```
where `k_block = k / 32` (block within the K dimension).

---

## 7. Force-Feeding EMA Scales at Save Time

### Problem with Natural Q4_0 Save

When saving loraA/loraB as Q4_0, the naive approach calls `quantize_q4_0(fp32_data)`
which recomputes block scales from `max_abs` on the *current* float values.

But the current float values of loraA/loraB have been shaped by the EMA-guided
fake-quant throughout training. The final max_abs of the float values may differ
slightly from the EMA scale. Recomputing from max_abs would throw away the calibrated
EMA information.

### Force-Feed

`build_q4_0_forced_blocks` writes the **EMA scale directly into the Q4_0 block
header** instead of recomputing it. The quantization formula then uses the EMA scale:

```
q_i = round(x_i / d_ema)    (using calibrated d_ema, NOT max_abs(block)/7)
```

After quantizing, `repack_q4_0` applies the standard XOR tiling for the GEMM kernel.

The result: the saved Q4_0 LoRA file encodes exactly the same quantization grid that
was used throughout training — perfect consistency between training and inference.

If QAT was not active (no EMA stats), `save_weight_lora_q4` falls back to natural
Q4_0 (recompute scale from max_abs).

```cpp
// transformer.cpp  save_weight_lora_q4()
if (LORA_QAT) {
    auto [a_bd, b_bd] = getRegisteredBlockScales(layer_name);
    q4_bytes = build_q4_0_forced_blocks(fp32_data, K, N, block_d_ema);
} else {
    q4_bytes = build_q4_0_natural(fp32_data, K, N);
}
```

---

## 8. W4A8 GEMM — Inference Kernel

At inference time, the model needs to compute:

```
y = x × W^T + scaling × x × loraA^T × loraB^T
```

where W, loraA, loraB are all stored as Q4_0 (4-bit), and x is FP32 activations.

This is a **W4A8** operation: 4-bit Weights × 8-bit Activations (x is quantized to
8-bit on-the-fly as `block_q8_0` before the GEMM).

### Path Selection

- **M=1** (single token generation): GEMV path (`__ggml_q4_0_4x8_q8_0_GEMV`)
- **M>1** (prefill, batch): GEMM path (`__ggml_q4_0_4x8_q8_0_GEMM`)

The `4x8` refers to the tile shape: 4 rows of the weight matrix processed
simultaneously × 8-element blocks of activations.

### Repacked Format Role

The repacked Q4_0 format (`block_q4_0x4`) interleaves 4 weight rows so the GEMM tile
can load them in a single cache-line-aligned read. The XOR-0x8 pre-processing means
the kernel works directly with signed nibbles without a runtime subtract.

### LoRA Path at Inference

At inference, loraA and loraB are also loaded as Q4_0 (from `lora_weights.q4.bin`).
The inference path runs three separate W4A8 GEMMs per FC layer:
1. `x × W^T` — base model output
2. `x × loraA^T` — bottleneck projection (r-dim)
3. `tmp × loraB^T` — expand back (N-dim)

Results are accumulated: `output = base_output + scaling × lora_output`

---

## 9. Training Pipeline

### 9.1 Training Data Format

The training binary `nntr_lora_train` supports two file formats:

**Plain format** (SST2 and similar classification tasks) — one line per sample:
```
{review or sentence text} Positive
{review or sentence text} Negative
```
The last token on the line (`Positive` / `Negative`) is the prediction target.
Everything before it is the input context. SST2 (`sst2_data/train.txt`) uses this
format: the model is trained to predict the sentiment label given the review text.

**Chat format** (LAMP and instruction tasks) — four lines per sample:
```
<|im_start|>user
{input question or review}<|im_end|>
<|im_start|>assistant
{expected answer}
```
The last line (e.g., `3` for a rating) is the prediction target.
Everything before it is the input context.

### 9.2 `loadTextFile` — Chat Format Detection

`lora_train.cpp::loadTextFile` auto-detects the format:

- **Plain format** (one line = one sample): `{text} Positive`
  → each line is tokenized separately, last token is the label (`Positive`/`Negative`)
  → used by SST2 (`train.txt`): straightforward sentiment classification
- **Chat format** (multiple lines = one sample): detected when any line starts with
  `<|im_start|>user`
  → lines are accumulated until the next `<|im_start|>user` marker, then joined
  with `\n` and tokenized as a single sample
  → used by LAMP-3 (`lamp3_user_train.txt`): rating prediction in chat context

This grouping for chat format is critical: without it, each line of a 4-line chat
exchange becomes a separate broken sample, training the model to predict structural
tokens (`user`, `assistant`, `<|im_end|>`) instead of the actual answer.

### 9.3 `dataCb` — Single-Token Prediction

`dataCb` implements a **next-single-token** prediction task:

```
input  = tokens[0 .. n-2]   (left-padded to seq_len)
label  = tokens[n-1]         (last token, one-hot over vocab_size=151936)
```

The input is left-padded so the last real token sits at position `seq_len-1` where
the LM head reads from. Right-padding would misalign the LM head read position.

For a plain SST2 sample:
```
This movie was absolutely wonderful and moving. Positive
──────────────────────────────────────────────  ↑ label
```
This trains: "given the review text, predict `Positive`."

For a chat-format LAMP-3 sample:
```
<|im_start|>user\nWhat is the score... review: {text}<|im_end|>\n<|im_start|>assistant\n3
──────────────────────────────────────────── input ──────────────────────────────────  ↑ label
```
This trains: "given the full chat context ending at `assistant\n`, predict `3`."
At inference, the model receives exactly that context — perfect match.

### 9.4 Training Loop

```
for each epoch:
  for each sample:
    1. Forward pass:
       a. input × W^T            (Q4_0 base, no gradient)
       b. fake_quant(loraA)       (EMA update if training)
       c. fake_quant(loraB)
       d. input × a_fq^T → tmp
       e. tmp × b_fq^T → lora_out
       f. output = base + scaling × lora_out
    2. Loss: cross-entropy(output[last_pos], label_token)
    3. Backward: gradients flow through b_fq, a_fq → loraB, loraA
       (STE: gradient passes through fake-quant unchanged)
    4. Adam optimizer updates loraA, loraB
  
  epoch_cb():
    - print training loss, validation loss, perplexity
    - print EMA scale stats (mean block scale per layer)
    - if val_loss improved: save checkpoint (lora_weights.bin + lora_weights.q4.bin)
    - else: patience countdown → early stop
```

### 9.5 Why Training Context Must Match Inference Context

The LoRA adapters only activate for token patterns they were trained on.

**SST2 plain format** sidesteps this issue because the task is coarse sentiment
classification — the LoRA pushes the model so strongly toward `Positive`/`Negative`
that it fires regardless of surrounding chat template tokens. This is essentially
mode collapse onto two tokens, which happens to work for binary classification.

**LAMP chat format** requires exact context matching. Qwen3's inference wraps every
prompt in `<|im_start|>user…<|im_end|>\n<|im_start|>assistant\n`. The LoRA must
fire at that exact context to influence the output. If training used plain format
instead, the adapter learns to fire on plain-text tokens that never appear at
inference — the adapter has zero effect despite having low training loss.

---

## 10. Inference Pipeline

### 10.1 Configuration

`nntr_config.json` controls inference mode:

| Field                | Effect                                           |
|----------------------|--------------------------------------------------|
| `lora_file_name`     | Load FP32 LoRA (loraA/loraB in FP32)            |
| `lora_q4_file_name`  | Load Q4_0 LoRA (triggers `lora_weight_q4=true`) |
| `lora_q6k_file_name` | Load Q6_K LoRA                                  |
| `sample_input`       | Default prompt (used when no argv[2] given)      |

### 10.2 Load Sequence (`main.cpp`)

```
1. Load config.json, generation_config.json, nntr_config.json
2. if lora_q4_file_name present: inject lora_weight_q4=true
   (this tells fc_layer to allocate Q4_0 tensors for loraA/loraB)
3. model->initialize()
   - allocates all weight tensors
   - Q4_0 loraA/loraB if lora_weight_q4=true
4. model->load_weight_lora_q4(base_file, lora_file)
   - loads Q4_0 base weights
   - loads Q4_0 LoRA weights into loraA/loraB slots
5. model->run(prompt)
   - apply chat template
   - tokenize
   - prefill (process prompt tokens)
   - autoregressive decode until EOS or max_length
```

### 10.3 Chat Template

Qwen3 wraps user input:
```
<|im_start|>user
{user prompt}<|im_end|>
<|im_start|>assistant
```
If the model has `<think>` capability enabled (Qwen3's reasoning mode), a `<think>`
block may precede the actual answer. The model generates tokens until `<|im_end|>`.

---

## 11. LAMP Personalization

### What is LAMP?

LAMP (Language Model Personalization) is a benchmark for adapting LLMs to individual
users. Each user has a history of past interactions (product reviews, articles read,
papers cited) that define their personal style.

### Per-User LoRA Approach

Instead of fine-tuning a global LoRA on all users, each user gets their own dedicated
LoRA adapter trained on their personal history. The adapter encodes the user's
preferences directly into the adapter weights.

```
User A's 90 past reviews → train LoRA_A → lora_weights_userA.q4.bin
User B's 90 past reviews → train LoRA_B → lora_weights_userB.q4.bin

At inference for user A: load base_model + LoRA_A → personalized predictions
At inference for user B: load base_model + LoRA_B → personalized predictions
```

### LaMP-3: Product Rating Task

The rating task (LaMP-3) asks the model to rate a product review on a 1–5 scale.

Training sample format:
```
<|im_start|>user
What is the score of the following review on a scale of 1 to 5? Just answer
with 1, 2, 3, 4, or 5 without further explanation. review: {review text}<|im_end|>
<|im_start|>assistant
{rating digit}
```

User 206515 (selected for proof-of-concept):
- 113 unique reviews, rating distribution: 1:10, 2:13, 3:30, 4:23, 5:37
- Average rating ~3.3 vs Amazon's typical ~4.5 positivity bias
- A LoRA trained on this user should predict lower ratings than the base model

---

## 12. Memory Breakdown

> **Note:** The numbers below combine analytical estimates (derived from model
> architecture constants: rank=32, 28 layers, 7 targets, hidden=1024, inter=2816)
> with **measured values from the completed LAMP-3 user 206515 training run**
> (20 epochs, QAT+Q4_0, Qwen3-0.6B). The `=== Memory Usage Summary ===` block
> printed at the end of each training run provides the snapshots and deltas.

### LoRA Memory: Analytical vs Measured (rank=32, 28 layers, 7 targets)

| Component               | Analytical estimate | Measured (LAMP-3 run) | How computed                              |
|-------------------------|---------------------|-----------------------|-------------------------------------------|
| LoRA weights (FP32)     | ~63 MB              | **66 MB**             | (loraA + loraB) × layers × 4 bytes        |
| LoRA gradients          | ~63 MB              | **66 MB**             | same shape as weights, one grad tensor each|
| Adam optimizer states   | ~126 MB             | **133 MB**            | 2 × weights (m and v per parameter)        |
| QAT fq tensors          | ~63 MB              | **66 MB**             | a_fq + b_fq FP32 copies (QAT only)        |
| **LoRA total (QAT)**    | ~315 MB             | **332 MB**            | weights + grads + Adam + fq                |
| **LoRA total (no QAT)** | ~252 MB             | **265 MB**            | weights + grads + Adam                     |

The `peak - pre_train` delta (513 MB measured) equals LoRA total (332 MB) + activation/gradient buffers (~181 MB). QAT adds exactly one extra set of fq tensors (66 MB) vs non-QAT.

### Total Training Memory (Qwen3-0.6B Q4_0) — Measured

| Stage                           | Measured          | Notes                                      |
|---------------------------------|-------------------|--------------------------------------------|
| Process baseline                | **12 MB**         | before any model allocation                |
| After model graph init          | **1158 MB** (+1145 MB) | all tensors allocated, weights not loaded yet |
| After base weights load         | **1159 MB** (+0 MB)    | Q4_0 weights embedded in graph at init     |
| Pre-train ready                 | **1254 MB** (+95 MB)   | tokenizer, data loader, optimizer init     |
| After epoch 1 (first backward)  | **1765 MB** (+511 MB)  | activations + all LoRA states materialized |
| **Peak during training (QAT)**  | **1767 MB** (+513 MB above pre-train) | peak activation+grad+optimizer overhead |
| After training done             | **1321 MB**       | optimizer states freed, activations freed  |

### Comparison: LoRA QAT vs Full Fine-Tuning (FP32)

| Approach            | Weights        | Gradients | Optimizer  | Total       |
|---------------------|----------------|-----------|------------|-------------|
| Full fine-tune FP32 | ~2400 MB       | ~2400 MB  | ~4800 MB   | ~9600 MB    |
| LoRA QAT (rank=32)  | ~730 + 66 MB   | 66 MB     | 133 MB     | **1767 MB** |
| **Reduction**       |                |           |            | **~82%**    |

---

## 13. Data Flow Diagrams

### Diagram A — PTQ: Quantizing the Base Model (one-time offline step)

```
FP32 Model Weights
(nntr_qwen3_0.6b_fp32.bin)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  nntr_quantize utility                                          │
│                                                                 │
│  For each FC layer weight matrix W [K×N]:                       │
│    1. Split into blocks of 32 elements along K dim             │
│    2. Per block: d = max_abs(block) / 7                        │
│    3. Encode: q_i = round(value_i / d) + 8  ∈ [0,15]          │
│    4. Pack 2 nibbles/byte → qs[16]                              │
│    5. repack_q4_0: interleave 4 rows + XOR nibbles with 0x88   │
│                                                                 │
│  Embedding layer → stays FP32                                   │
│  LM head         → stays FP32                                   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Q4_0 Quantized Model
(nntr_qwen3_0.6b_q40_embdfp32.bin)
~730 MB  (vs ~2400 MB FP32)
```

---

### Diagram B — QAT LoRA Training (per fine-tuning run)

```
Training Data File (sst2_data/train.txt — plain format)
────────────────────────────────────────────────────────
"This movie was absolutely wonderful and moving. Positive"
"The acting was poor and the plot made no sense. Negative"
  ← one line = one sample, no chat template needed
                │
                ▼ tokenize (each line as a single string)
[token_ids for review text ..., token_id for "Positive"]
                │
                │ dataCb splits:
                ├─ input  = tokens[0..-2] (left-padded to seq_len=128)
                └─ label  = token[-1] = "Positive" or "Negative"
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  FORWARD PASS  (FullyConnectedLayer::forwarding)                   │
│                                                                    │
│  ┌─────────────────────────────┐   ┌────────────────────────────┐ │
│  │ Base Model Path (FROZEN)    │   │ LoRA Path (TRAINED)        │ │
│  │                             │   │                            │ │
│  │ input × W_q4^T              │   │ a_fq = fakeQuant(loraA)    │ │
│  │   W_q4: repacked Q4_0       │   │   EMA d updated per block  │ │
│  │   W4A8 GEMM kernel          │   │ b_fq = fakeQuant(loraB)    │ │
│  │   output: FP32 activations  │   │ tmp  = input × a_fq^T      │ │
│  │                             │   │ out  = tmp   × b_fq^T      │ │
│  └──────────────┬──────────────┘   └──────────────┬─────────────┘ │
│                 │                                  │               │
│                 └──────────── + scaling ───────────┘               │
│                                    │                               │
│                                    ▼                               │
│                            layer output (FP32)                     │
└────────────────────────────────────┬───────────────────────────────┘
                                     │ (through 28 transformer layers)
                                     ▼
                              LM Head (FP32)
                                     │
                                     ▼
                          logits over vocab (151936 classes)
                                     │
                                     ▼
                   CrossEntropy Loss vs label token "Positive"/"Negative"
                                     │
┌────────────────────────────────────▼───────────────────────────────┐
│  BACKWARD PASS                                                     │
│                                                                    │
│  gradient flows through LM Head → transformer layers              │
│                                                                    │
│  At each FC layer:                                                 │
│    d_loss/d_loraB ← gradient through b_fq (STE: passes through)   │
│    d_loss/d_loraA ← gradient through a_fq (STE: passes through)   │
│    d_loss/d_W     = 0  (W is frozen, trainable=false)             │
│                                                                    │
│  Adam optimizer: update loraA, loraB with lr=1e-5                  │
└────────────────────────────────────────────────────────────────────┘
                          │
              (repeat for all 90 samples × 10 epochs)
                          │
                          ▼
            Early stopping: if val_loss improves →
┌─────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT SAVE                                                    │
│                                                                    │
│  save_weight_lora():                                               │
│    → lora_weights.bin (FP32 loraA/loraB for all layers)           │
│                                                                    │
│  save_weight_lora_q4():                                            │
│    for each layer:                                                 │
│      if QAT active:                                                │
│        read EMA scales from s_block_d_registry[layer_name]        │
│        build_q4_0_forced_blocks(fp32_data, K, N, d_ema)           │
│          → encode nibbles using d_ema (not max_abs)               │
│      else:                                                         │
│        build_q4_0_natural(fp32_data, K, N)                        │
│      repack_q4_0 → write to lora_weights.q4.bin                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Diagram C — Q4_0 LoRA Inference

```
User Prompt: "Rate this review: The product broke after one day."
                    │
                    ▼
         Qwen3 Chat Template Applied
         ─────────────────────────────
         <|im_start|>user
         Rate this review...<|im_end|>
         <|im_start|>assistant
                    │
                    ▼ tokenize
              [token_ids ...]
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│  LOAD SEQUENCE (main.cpp)                                        │
│                                                                  │
│  nntr_config.json has lora_q4_file_name → inject lora_weight_q4 │
│                                                                  │
│  model->initialize()  allocates Q4_0 tensors for loraA, loraB   │
│                                                                  │
│  load_weight_lora_q4(base_file, lora_file):                      │
│    base_file → W tensors in Q4_0 repacked format                 │
│    lora_file → loraA, loraB in Q4_0 repacked format              │
└───────────────────┬──────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│  PER-LAYER FC COMPUTATION (incremental_forwarding, M=1)           │
│                                                                   │
│  1. base:  x_q8 = quantize_q8_0(input_fp32)    (on-the-fly)      │
│            y_base = W_q4 × x_q8  (W4A8 GEMV kernel)              │
│                                                                   │
│  2. lora:  tmp = loraA_q4 × x_q8               (W4A8 GEMV)       │
│            y_lora = loraB_q4 × tmp_q8           (W4A8 GEMV)       │
│                                                                   │
│  3. merge: output = y_base + scaling × y_lora                    │
└───────────────────┬───────────────────────────────────────────────┘
                    │ (28 layers)
                    ▼
               LM Head → logits
                    │
                    ▼ sample (greedy/top-p)
                next token
                    │
              ┌─────┴─────┐
              │ EOS?       │ No → feed back as next input
              └─────┬─────┘
                    │ Yes
                    ▼
              decoded text output
              e.g. "1" or "2"  (personalized low rating)
```

---

### Diagram D — LAMP Per-User Personalization

```
LaMP-3 Dataset
──────────────
User 206515: 113 unique product reviews with ratings 1–5
  rating distribution: 1:10, 2:13, 3:30, 4:23, 5:37
  (more critical than average Amazon reviewer)

                    │
                    ▼ lamp3_user_convert.py
                    
lamp3_user_train.txt (90 samples, 80% split)
lamp3_user_test.txt  (23 samples, 20% split)
                    │
                    ▼ nntr_lora_train
                    
LoRA adapter trained on user 206515's rating style
→ lora_weights.bin + lora_weights.q4.bin

                    │
         ┌──────────┴──────────┐
         ▼                     ▼
  Base Qwen3 alone       Base Qwen3 + User LoRA
  "The shoe broke"  →    "The shoe broke"  →
  predicts: "4"          predicts: "2"
  (optimistic prior)     (calibrated to this user)
```

---

## 14. File Reference

| File | Role |
|------|------|
| `nntrainer/layers/fc_layer.h` | LoRA tensor declarations, EMA state, fakeQuant signature, static registries |
| `nntrainer/layers/fc_layer.cpp` | `forwarding`: base+LoRA path, `fakeQuantizeQ4_0`: EMA + STE |
| `nntrainer/tensor/quantizer.cpp` | `GgmlQuantizer::quantize/dequantize`, `quantize_q4_0`, `repack_q4_0`, `unpack_q4_0` |
| `nntrainer/tensor/q4_0_tensor.h/cpp` | Q4_0 tensor type, `getData()` returns raw byte buffer |
| `Applications/CausalLM/models/transformer.cpp` | `save_weight_lora_q4`: force-feed EMA scales; `build_q4_0_forced_blocks`; `load_weight_lora_q4` |
| `Applications/CausalLM/train_qwen3_lora_master.cpp` | Training entry point, epoch callback, early stopping, memory stats |
| `Applications/CausalLM/lora_train.cpp` | `loadTextFile`: chat-format grouping; `dataCb`: single-token prediction |
| `Applications/CausalLM/main.cpp` | Inference entry point, `lora_q4_file_name` injection |
| `Applications/CausalLM/sst2_data/zephyr_chat.txt` | Identity fine-tuning: chat-format, single-word targets |
| `Applications/CausalLM/sst2_data/lamp3_user_train.txt` | LAMP-3 user 206515 training split (90 samples) |
| `Applications/CausalLM/sst2_data/lamp3_user_test.txt` | LAMP-3 user 206515 test split (23 samples) |
| `Applications/CausalLM/sst2_data/lamp3_user_convert.py` | Converts LAMP HuggingFace dataset → chat-format training files |
| `res/qwen3/qwen3-0.6b/nntr_config.json` | Model config: LoRA rank/alpha/targets, weight filenames, sample prompt |

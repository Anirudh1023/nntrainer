# Multi-Batch Inference Fix — NNTrainer CausalLM
**Branch:** `mb` | **Commit:** `53f92a8f`

---

## Background: What is "Multi-Batch Inference"?

Imagine you want an AI model to answer two different questions at the same time, instead of one after the other. That's what multi-batch inference is. Instead of processing 1 prompt sequentially, you feed N prompts simultaneously (a "batch" of N) to the model.

**The benefit:** The model's weights (its "brain", ~3 GB) are loaded from memory once and shared across all N prompts. So you pay the memory cost only once, but get N results. This is the same trick that makes cloud AI services so efficient — they batch hundreds of user requests together.

Before this fix, NNTrainer's CausalLM **crashed** whenever `batch_size > 1` was set in `nntr_config.json`. This document explains every change we made to fix it.

---

## How Memory Works in This Codebase (Important Context)

Before diving into bugs, you need to understand how NNTrainer stores numbers.

Think of a **Tensor** as a rectangular box of numbers with 4 dimensions:
```
Tensor shape: (Batch, Channel, Height, Width)
              (  2,      1,    1024,  1536 )
```
- **Batch** = how many independent items are processing at once
- **Height** = sequence length (number of tokens)
- **Width** = embedding dimension (how many numbers represent each token)

When `batch_size = 2`, the tensor is like two floors of a building stacked on top of each other. To find where Batch 1's data starts, you skip past all of Batch 0's data. The number of elements in one batch's "floor" is called `getFeatureLen()`:

```
getFeatureLen() = Channel × Height × Width = 1 × 1024 × 1536 = 1,572,864 elements
```

Most bugs in this fix came from code using the **wrong stride** to jump between batches.

---

## File 1: `causal_lm.h`

### Change: `pending_ids_` — Per-batch token accumulation buffer

**What is `pending_ids_`?**

When the AI generates a word, it produces a number (a "token ID"). But some words in Unicode take multiple tokens to spell out. For example, the word "Hello" might be tokenized as two pieces: `[123, 456]`. Only when you decode both pieces together do you get "Hello".

`pending_ids_` is a small scratch buffer that holds these token pieces until they form a complete word.

**The Bug:**

```cpp
// BEFORE (broken for batch > 1)
std::vector<int> pending_ids_;
```

This was ONE shared list for ALL batches. Batch 0 would push its half-formed token in, then Batch 1 would push its half-formed token in, and then the decoder would try to interpret both tokens as one word — nonsense!

**The Fix:**

```cpp
// AFTER (each batch has its own independent list)
std::vector<std::vector<int>> pending_ids_;
```

Now it's a list of lists — one inner list per batch. Batch 0 has its own slot, Batch 1 has its own slot. They never interfere.

**Memory impact:** Completely negligible. Each inner list holds at most 3 integers (~12 bytes). For batch_size=2, total extra cost is ~50 bytes out of a 3 GB model.

---

## File 2: `causal_lm.cpp`

This is the main orchestrator file. It had the most bugs.

### Change 1: Initialize per-batch buffers in `setupParameters()`

**Before:**
```cpp
for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");
// pending_ids_ was NOT initialized here
```

**After:**
```cpp
for (unsigned int i = 0; i < BATCH_SIZE; ++i) {
    output_list.push_back("");
    pending_ids_.push_back(std::vector<int>()); // one empty list per batch
}
```

We added initialization so that `pending_ids_[0]`, `pending_ids_[1]`, etc. actually exist before we try to use them.

---

### Change 2: Reset per-batch state at the start of `run()`

When you call `run()` a second time (e.g., new conversation), stale data from the previous run must be cleared.

**Before:** Only `output_list` was cleared. `pending_ids_` was left with garbage from the last run.

**After:**
```cpp
output_list.clear();
pending_ids_.clear();
for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
    pending_ids_.push_back(std::vector<int>());
}
```

---

### Change 3: Input tensor packing stride — The most critical bug

**Background:** The AI reads your prompt as a list of token numbers (e.g., `[1, 523, 4402, 89]`). These are placed into an array called `input_sample` which the first layer of the model reads.

For multi-batch, you place Batch 0's tokens first, then Batch 1's tokens after a gap. The question is: **how big is the gap?**

The gap must match exactly the height of the input tensor registered with the model. The input layer was registered with `height = INIT_SEQ_LEN = 1024`. So the gap between batches is `1 × 1024` elements.

**The Bug:** The code was using `MAX_SEQ_LEN = 2048` as the gap instead:

```cpp
// BEFORE — wrong gap, puts Batch 1 at position 2048
input_sample[b * MAX_SEQ_LEN + i] = token;
```

Batch 0's tokens were at positions 0–17 ✅  
Batch 1's tokens were at positions 2048–2065 ❌ (the model only sees up to position 1023!)

When the Embedding layer tried to look up token at index 2048, it found `0.0` (uninitialized memory) and interpreted that as token ID `0`, which was then out of range → **crash: "input word index is greater than in_dim"**.

**The Fix:**
```cpp
// AFTER — correct gap matches the tensor stride
input_sample[b * INIT_SEQ_LEN + i] = token;
```

This same fix was applied in **3 places** in the generation loop:
1. Prefill packing (before the first inference)
2. First decode token packing (after prefill returns)
3. Subsequent decode token packing (every generation step)

---

### Change 4: Prefill token selection — Replaced `generate_multi_tokens`

After the prefill (reading your full prompt), the model outputs a probability distribution over all vocabulary tokens. We need to pick the most likely next token for each batch item.

**Before:** The code called `generate_multi_tokens(output[0], NUM_VOCAB, BATCH_SIZE, ...)`. This function was designed to pick the **top-N tokens from a single flat probability list** — treating it as if all batches shared one output. For batch_size=2 it would pick the 2 highest probability tokens from Batch 0's output instead of picking 1 token each from Batch 0 and Batch 1.

**After:**
```cpp
std::vector<unsigned int> id_list;
for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    float *batch_logits = output[0] + b * NUM_VOCAB;
    unsigned int argmax = std::distance(
        batch_logits, std::max_element(batch_logits, batch_logits + NUM_VOCAB));
    id_list.push_back(argmax);
}
```

This correctly finds the most likely token **independently for each batch** by jumping to each batch's section of the output buffer (`b * NUM_VOCAB`).

---

### Change 5: `registerOutputs` — Per-batch decoding + live streaming

**Before (buggy):**
```cpp
pending_ids_.push_back(ids[b]);          // ALL batches write to same list
std::string decoded_str = tokenizer->Decode(pending_ids_); // decodes mix of all batches
if (decoded_str.back() == ',') ...        // CRASH if decoded_str is empty!
```

**After (correct):**
```cpp
pending_ids_[b].push_back(ids[b]);       // each batch writes to its own slot
std::string decoded_str = tokenizer->Decode(pending_ids_[b]); // decode only this batch
if (decoded_str.empty()) { /* wait */ }   // safe empty check added
else if (decoded_str.back() == ',') ...   // only reached if non-empty
...
// Batch 0 streams live to screen; other batches accumulate silently
if (log_output && b == 0) {
    std::cout << decoded_str;
}
output_list[b].append(decoded_str);
pending_ids_[b].clear();
```

The empty string guard was essential — when a token generates an empty decoded string (special tokens, incomplete multi-byte sequences), calling `.back()` on an empty string is undefined behaviour in C++ and caused a crash.

---

### Change 6: End-of-run output display

**Before:** Output was printed character-by-character to stdout during generation (worked fine for batch=1 but caused interleaved doubled output for batch > 1).

**After:** Batch 0 still streams live (so the user sees output being generated in real time). When generation finishes, any additional batches are printed with clear labels:

```
<live streaming of Batch 0 during generation>
...

--- [Batch 0 was streamed above] ---
--- [Batch 1] ---
A large language model is...

--- [Batch 2] ---
A large language model (LLM) is...
```

---

## File 3: `fc_layer.cpp`

### Change: Correct input tensor offset in `incremental_forwarding`

**What is an FC (Fully Connected) Layer?**

An FC layer is a translation machine. It takes an input vector of size M and produces an output vector of size N using a weight matrix. In the Qwen3 model, the Q-projection FC layer takes a 1536-wide input and produces a 2048-wide output.

**The Bug:**

When looping over batches, the code needed to find where each batch's **input** data starts. It was computing this using the **output** tensor's feature length instead of the **input** tensor's feature length:

```cpp
// BEFORE — wrong: uses output (hidden) stride for the input
Tensor input_step = input_.getSharedDataTensor(
    input_step_dim, b * hidden_dim.getFeatureLen(), true);
```

For the Q-projection layer with batch_size=2:
- `input_dim.getFeatureLen() = 1024 × 1536 = 1,572,864` (correct)
- `hidden_dim.getFeatureLen() = 1024 × 2048 = 2,097,152` (wrong, too large)

For Batch 1 (`b=1`): the code tried to read from memory position `2,097,152`. But the input tensor only has `2 × 1,572,864 = 3,145,728` total elements. Adding the slice size `(1 × 1024 × 1536 = 1,572,864)`:

```
2,097,152 + 1,572,864 = 3,670,016 > 3,145,728  →  CRASH
```

**The Fix:**
```cpp
// AFTER — correct: uses input stride for input, output stride for output
Tensor input_step = input_.getSharedDataTensor(
    input_step_dim, b * input_dim.getFeatureLen(), true);  // ← changed
Tensor hidden_step = hidden_.getSharedDataTensor(
    hidden_step_dim, b * hidden_dim.getFeatureLen(), true); // ← unchanged
```

One word change. One character of difference. Prevents a memory out-of-bounds crash.

---

## File 4: `mha_core.cpp` — Two bugs in the Attention layer

The Multi-Head Attention (MHA) layer is the "brain" of the transformer. It maintains a KV Cache — a memory of everything the model has seen so far.

### Bug 1: Wrong batch dimension on KV cache slices

**What are KV cache slices?**

The full KV cache stores memories for ALL batches: shape `(2, 1, 1536, 1024)`.  
When processing Batch 1, we need a "window" into just Batch 1's portion: shape `(1, 1, cache_to, 1024)`.

**The Bug:** When creating this window, the code inherited the full tensor's `batch=2` dimension instead of forcing it to `batch=1`:

```cpp
// BEFORE — batch=2 inherited, requests 2x too much memory
ml::train::TensorDim cached_key_dim = cache_key_dim; // batch=2 !
cached_key_dim.height(cache_to);

Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
// For batch=1: offset = 1×featureLen, slice = 2×cache_to×width
// Total = featureLen + 2×cache_to×width > total tensor size → CRASH
```

**The Fix:**
```cpp
// AFTER — force batch=1 before using as per-batch slice
ml::train::TensorDim cached_key_dim = cache_key_dim;
cached_key_dim.batch(1);   // ← added
cached_key_dim.height(cache_to);
// Now: offset = 1×featureLen, slice = 1×cache_to×width → fits safely
```

Applied to both `one_batch_incremental_forwarding` overloads.

---

### Bug 2: KV cache being shrunk during decode steps

**Background:**

The model works in two phases:
1. **Prefill**: Reads your full prompt (e.g., 18 tokens at once). KV cache is sized to `18 + 512 = 530` slots.
2. **Decode**: Generates one new token at a time. Input height = 1.

There is a function `updateTensorsByInputDimensions` that resizes tensors when the input size changes. It recalculates `max_timestep` as `input_height + max_new_tokens`.

**The Bug:** When decode starts, `input_height = 1`, so it computed:
```
max_timestep = 1 + 512 = 513
```
And then **resized the KV cache down to 513 slots**. But the cache already contained 18 prefill tokens' memories! Shrinking it destroyed those memories AND made the cache too small for Batch 1 to safely access, causing crashes.

**The Fix:**
```cpp
// BEFORE — always resizes, even during decode
max_timestep = height + max_new_tokens;

// AFTER — only resize during prefill (height > 1)
if (height > 1) {
    max_timestep = height + max_new_tokens;
}
// During decode (height == 1), keep the existing max_timestep intact
```

---

## Summary Table

| # | File | What Changed | Why It Was Needed |
|---|------|-------------|-------------------|
| 1 | `causal_lm.h` | `pending_ids_`: `vector<int>` → `vector<vector<int>>` | Shared token buffer caused cross-batch decoding corruption |
| 2 | `causal_lm.cpp` | Initialize `pending_ids_` in setup + run reset | Vector must be pre-sized before indexing by batch |
| 3 | `causal_lm.cpp` | `input_sample` stride: `MAX_SEQ_LEN` → `INIT_SEQ_LEN` | Tensor stride is INIT_SEQ_LEN (1024), not MAX_SEQ_LEN (2048) |
| 4 | `causal_lm.cpp` | Replaced `generate_multi_tokens` with per-batch argmax | Old function picked top-N from one batch, not 1 from each batch |
| 5 | `causal_lm.cpp` | Per-batch `registerOutputs` + empty string guard | Prevented `.back()` crash on empty decoded strings |
| 6 | `causal_lm.cpp` | Stream batch 0 live; print others at end | Removed interleaved doubled output |
| 7 | `fc_layer.cpp` | Input offset: `hidden_dim.getFeatureLen()` → `input_dim.getFeatureLen()` | Input/output widths differ in Q/K/V projections causing OOB |
| 8 | `mha_core.cpp` | `cached_key/value_dim.batch(1)` before slicing | Without this, each per-batch slice requested 2× the needed memory |
| 9 | `mha_core.cpp` | `if (height > 1)` guard in `updateTensorsByInputDimensions` | Prevented KV cache from being shrunk and corrupted during decode |

---

## What Was NOT Changed

- No changes to any compute kernels (NEON, BLAS, GGML)
- No changes to model architecture or weight loading
- No changes to tokenizer or vocabulary  
- No changes to attention math, RoPE, softmax, or RMS norm logic
- `qkv_layer.cpp` was modified locally but **not committed** — it is unused by Qwen3

## Performance Impact

| Scenario | Impact |
|----------|--------|
| batch_size=1 | Zero. All batch offsets evaluate to `0 × stride = 0`. Identical code path. |
| batch_size=N | Correct but sequential. Each batch's GEMM runs one after another (existing `// @todo parallelize` in fc_layer.cpp). Not worse than before — it previously crashed instead. |
